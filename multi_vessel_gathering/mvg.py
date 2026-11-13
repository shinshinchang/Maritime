#!/usr/bin/env python3
"""AIS multi-vessel gathering Week 10 prototype. Read-only input, disk-backed staging."""
from __future__ import annotations
import argparse, csv, gzip, json, math, sqlite3, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ALIASES = {
    'vessel': ['mmsi', 'vessel_mmsi', 'ship_mmsi', 'maritime_mobile_service_identity'],
    'lat': ['lat', 'latitude', 'position_latitude', 'poslat'],
    'lon': ['lon', 'lng', 'longitude', 'position_longitude', 'poslon'],
    'time': ['record_time', 'posdt', 'pos_dt', 'position_time', 'position_datetime', 'timestamp', 'base_datetime', 'datetime', 'time'],
    'speed': ['sog', 'speed_over_ground', 'speed'],
}
EARTH_KM = 6371.0088
NM_TO_KM = 1.852

def normalized(s):
    return ''.join(c for c in str(s).casefold() if c.isalnum())

def discover_files(paths):
    results=[]
    for raw in paths:
        p=Path(raw)
        if p.is_dir():
            results += [f for f in p.rglob('*') if f.is_file() and (f.name.lower().endswith('.csv') or f.name.lower().endswith('.csv.gz'))]
        elif p.is_file():
            results.append(p)
        else:
            raise FileNotFoundError(f'Cannot find input: {p}')
    files=sorted(set(results))
    if not files: raise ValueError('No .csv or .csv.gz found')
    return files

def select_columns(columns, explicit):
    lookup={normalized(c):c for c in columns}
    result={}
    for key in ALIASES:
        if explicit.get(key):
            specified=explicit[key]
            match=lookup.get(normalized(specified))
            if match is None: raise ValueError(f'Column {specified!r} not found. Available: {list(columns)}')
            result[key]=match
        else:
            result[key]=next((lookup[normalized(s)] for s in ALIASES[key] if normalized(s) in lookup),None)
    missing=[k for k in ['vessel','lat','lon','time'] if result[k] is None]
    if missing: raise ValueError(f'Missing required fields {missing}. Available columns: {list(columns)}. Pass --{missing[0]}-col when appropriate. Do not treat src_id as a vessel ID without checking its semantics.')
    return result

def inspect(paths, args):
    report=[]
    for f in discover_files(paths):
        try:
            df=pd.read_csv(f,nrows=args.sample_rows,encoding=args.encoding,sep=args.sep,low_memory=False)
            names=list(df.columns)
            mapped=select_columns(names, columns_overrides(args))
            small=df[[c for c in mapped.values() if c]].head(5).copy()
            # Mask identifiers: do not print identifiable vessel IDs in diagnostic output
            if mapped['vessel'] in small: small[mapped['vessel']]='[vessel id]'
            report.append({'file':str(f),'size_mb':round(f.stat().st_size/1048576,2), 'columns':names,'mapped':mapped,'sample_rows':len(df),'null_counts_in_sample':{k:int(df[v].isna().sum()) for k,v in mapped.items() if v},'masked_example_rows':small.astype(str).to_dict('records')})
        except Exception as e:
            report.append({'file':str(f),'error':str(e)})
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.output: Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')

def columns_overrides(args):
    return {k:getattr(args, k+'_col',None) for k in ALIASES}

def parse_timestamps(series):
    # Auto-detect epoch seconds/milliseconds for purely numeric columns
    asnum=pd.to_numeric(series,errors='coerce')
    numeric_fraction=float(asnum.notna().mean()) if len(asnum) else 0
    if numeric_fraction>.95:
        median=asnum.abs().median()
        if median>1e11: return pd.to_datetime(asnum,unit='ms',utc=True,errors='coerce')
        if median>1e9: return pd.to_datetime(asnum,unit='s',utc=True,errors='coerce')
    return pd.to_datetime(series,utc=True,errors='coerce',format='mixed')

def db_open(path):
    conn=sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('CREATE TABLE IF NOT EXISTS points (bucket INTEGER NOT NULL, vessel TEXT NOT NULL, ts INTEGER NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, sog REAL, gx INTEGER NOT NULL, gy INTEGER NOT NULL, PRIMARY KEY (bucket, vessel)) WITHOUT ROWID')
    return conn

def load_files(conn, files, args):
    stats=Counter(); start=time.time(); field_map=None
    for f in files:
        print(f'Loading {f.name} ...',flush=True)
        # Header-only inspection: 109-column vendor CSV needs only 5 fields for Week 10.
        headers=list(pd.read_csv(f,nrows=0,encoding=args.encoding,sep=args.sep).columns)
        mapped=select_columns(headers, columns_overrides(args))
        required_columns=set(v for v in mapped.values() if v is not None)
        kwargs=dict(chunksize=args.chunk_rows,encoding=args.encoding,sep=args.sep,
                    usecols=lambda col: col in required_columns,
                    dtype={mapped['vessel']:'string'},low_memory=False)
        print(f'  selected {len(required_columns)} / {len(headers)} columns: {sorted(required_columns)}',flush=True)
        for frame in pd.read_csv(f,**kwargs):
            stats['rows_read']+=len(frame)
            if field_map is None: field_map=mapped
            df=pd.DataFrame({k:frame[v] if v else np.nan for k,v in mapped.items()})
            df['lat']=pd.to_numeric(df['lat'],errors='coerce')
            df['lon']=pd.to_numeric(df['lon'],errors='coerce')
            df['speed']=pd.to_numeric(df['speed'],errors='coerce')
            dt=parse_timestamps(df['time'])
            vessels=df['vessel'].astype('string').str.strip().str.replace(r'\.0$','',regex=True)
            good=(vessels.notna() & (vessels!='') & (~vessels.str.lower().isin(['nan','none','null'])) & dt.notna() & df['lat'].between(-90,90) & df['lon'].between(-180,180) & ~((df['lat']==0)&(df['lon']==0))).fillna(False)
            stats['invalid_rows']+=int((~good).sum())
            if not good.any(): continue
            subset=df.loc[good,['lat','lon','speed']].copy()
            subset['vessel']=vessels.loc[good].astype(str).values
            subset['ts']=(dt.loc[good].astype('int64')//1_000_000_000).values
            subset['bucket']=(subset['ts']//(args.time_minutes*60)).astype('int64')
            subset['gx']=np.floor(subset['lon']/args.grid_deg).astype('int64')
            subset['gy']=np.floor(subset['lat']/args.grid_deg).astype('int64')
            subset=subset.sort_values('ts').drop_duplicates(['bucket','vessel'],keep='last')
            rows=[(int(r.bucket),str(r.vessel),int(r.ts),float(r.lat),float(r.lon),None if pd.isna(r.speed) else float(r.speed),int(r.gx),int(r.gy)) for r in subset.itertuples(index=False)]
            conn.executemany('''INSERT INTO points(bucket,vessel,ts,lat,lon,sog,gx,gy) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(bucket,vessel) DO UPDATE SET ts=excluded.ts,lat=excluded.lat,lon=excluded.lon,sog=excluded.sog,gx=excluded.gx,gy=excluded.gy WHERE excluded.ts >= points.ts''',rows)
            conn.commit()
            stats['valid_rows']+=int(good.sum())
            if stats['rows_read']%(args.chunk_rows*10)<len(frame): print(f'  read {stats["rows_read"]:,} rows, elapsed {time.time()-start:.0f}s',flush=True)
    stats['retained_unique_vessel_time_bins']=conn.execute('SELECT COUNT(*) FROM points').fetchone()[0]
    return dict(stats), field_map

def run_eda(conn, out, ing, colmap, args):
    row=conn.execute('SELECT COUNT(*),COUNT(DISTINCT vessel),COUNT(DISTINCT bucket),MIN(ts),MAX(ts),MIN(lat),MAX(lat),MIN(lon),MAX(lon) FROM points').fetchone()
    summary={'ingestion':ing,'detected_columns':colmap,'unique_vessel_time_bins':row[0],'unique_vessels':row[1],'time_bins':row[2], 'time_min_utc':pd.to_datetime(row[3],unit='s',utc=True).isoformat() if row[3] is not None else None,'time_max_utc':pd.to_datetime(row[4],unit='s',utc=True).isoformat() if row[4] is not None else None,'latitude_range':row[5:7],'longitude_range':row[7:9],'settings':{'time_minutes':args.time_minutes,'grid_deg':args.grid_deg,'eps_nm':args.eps_nm,'min_vessels':args.min_vessels}, 'limitations':['Time with no timezone is interpreted as UTC; confirm posDt semantics','One most recent record per vessel per fixed time bin','Candidates are not verified anomalies','No continuity, port exclusions, or collision-course analysis in Week 10','Source is vendor scored_reports CSV; rule flags are not used as verified gathering labels']}
    (out/'eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    for name,sql,head in [
        ('top_vessels.csv','SELECT vessel,COUNT(*) AS time_bins FROM points GROUP BY vessel ORDER BY time_bins DESC LIMIT 50',['vessel','time_bins']),
        ('top_spatial_cells.csv','SELECT gx,gy,COUNT(*) AS records,COUNT(DISTINCT vessel) AS unique_vessels FROM points GROUP BY gx,gy ORDER BY records DESC LIMIT 100',['gx','gy','records','unique_vessels']),
        ('time_bin_counts.csv','SELECT bucket,COUNT(*) AS vessel_count FROM points GROUP BY bucket ORDER BY bucket',['bucket','vessel_count'])]:
        with (out/name).open('w',newline='',encoding='utf8') as fp:
            wr=csv.writer(fp);wr.writerow(head);wr.writerows(conn.execute(sql))
    return summary

def candidate_indices(points,min_vessels):
    cells={}
    for i,p in enumerate(points): cells.setdefault((p[4],p[5]),[]).append(i)
    selected=set()
    for (x,y),idx in cells.items():
        neighbors=[j for dx in [-1,0,1] for dy in [-1,0,1] for j in cells.get((x+dx,y+dy),[])]
        if len(neighbors)>=min_vessels: selected.update(idx)
    return sorted(selected)

def detect_bucket(points,args):
    idx=candidate_indices(points,args.min_vessels)
    if len(idx)<args.min_vessels: return []
    coords=np.radians(np.array([[points[i][1],points[i][2]] for i in idx],dtype=float))
    label=DBSCAN(eps=(args.eps_nm*NM_TO_KM)/EARTH_KM,min_samples=args.min_vessels,metric='haversine',algorithm='ball_tree').fit_predict(coords)
    clusters=[]
    for lab in sorted(set(label)-{-1}):
        inds=[idx[i] for i,v in enumerate(label) if v==lab]
        group=[points[i] for i in inds]
        # DBSCAN permits chain formation; record diameter, allow optional compactness threshold
        lat=np.radians(np.array([p[1] for p in group]));lon=np.radians(np.array([p[2] for p in group]))
        maxkm=0.;sumkm=0.;num=0
        for i in range(len(group)):
            dlat=lat[i+1:]-lat[i]; dlon=lon[i+1:]-lon[i]
            a=np.sin(dlat/2)**2+np.cos(lat[i])*np.cos(lat[i+1:])*np.sin(dlon/2)**2
            distances=2*EARTH_KM*np.arcsin(np.sqrt(np.clip(a,0,1)))
            if len(distances): maxkm=max(maxkm,float(np.max(distances)));sumkm+=float(np.sum(distances));num+=len(distances)
        if args.max_diameter_nm is not None and maxkm/NM_TO_KM>args.max_diameter_nm: continue
        speeds=[p[3] for p in group if p[3] is not None and math.isfinite(p[3])]
        clusters.append({'vessels':sorted(p[0] for p in group),'count':len(group),'lat':float(np.mean(np.degrees(lat))),'lon':float(np.mean(np.degrees(lon))),'diameter_nm':maxkm/NM_TO_KM,'mean_pair_distance_nm':sumkm/num/NM_TO_KM if num else 0.,'mean_sog_kn':float(np.mean(speeds)) if speeds else None,'group':group})
    return clusters

def run_detection(conn,out,args):
    outpath=out/'gathering_candidates.csv';top=[];total=0;last=None
    header=['candidate_id','time_bin_start_utc','group_size','vessel_ids','centroid_lat','centroid_lon','group_diameter_nm','mean_pair_distance_nm','mean_sog_kn','is_persistent','is_verified_anomaly']
    with outpath.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(header)
        cursor=conn.execute('SELECT bucket,vessel,lat,lon,sog,gx,gy FROM points ORDER BY bucket')
        import itertools
        for bucket,group_rows in itertools.groupby(cursor,key=lambda r:r[0]):
            points=[r[1:] for r in group_rows]
            if args.max_vessels_per_bin and len(points)>args.max_vessels_per_bin:
                print(f'WARNING: skipped bucket {bucket} with {len(points)} vessels over safety cap',file=sys.stderr)
                continue
            clusters=detect_bucket(points,args)
            for c in clusters:
                total+=1
                when=pd.to_datetime(bucket*args.time_minutes*60,unit='s',utc=True).isoformat()
                writer.writerow([f'G{total:07d}',when,c['count'],';'.join(c['vessels']),round(c['lat'],7),round(c['lon'],7),round(c['diameter_nm'],4),round(c['mean_pair_distance_nm'],4),round(c['mean_sog_kn'],3) if c['mean_sog_kn'] is not None else '',False,False])
                if len(top)<args.max_plot_candidates: top.append((when,c))
            if last is None or bucket-last>=288: print(f'  scanned through {pd.to_datetime(bucket*args.time_minutes*60,unit="s",utc=True).date()}, candidates={total}',flush=True);last=bucket
    return total,top

def plot_candidates(top,out):
    if not top:return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # No coastline base layer: explicitly scatter-only visualization
    fig,ax=plt.subplots(figsize=(9,6))
    for when,c in top:
        for p in c['group']: ax.scatter(p[2],p[1],s=18,alpha=.65)
        ax.scatter(c['lon'],c['lat'],marker='x',s=60,color='black')
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=f'Week 10 candidate locations (first {len(top)} clusters)')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'candidate_scatter.png',dpi=160);plt.close(fig)
    return True

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    files=discover_files(args.inputs)
    dbpath=out/'ais_staging.sqlite'
    if dbpath.exists() and not args.rebuild: raise SystemExit(f'{dbpath} already exists. Use a new --out directory or --rebuild to replace the staging data.')
    if args.rebuild and dbpath.exists(): dbpath.unlink()
    conn=db_open(dbpath)
    try:
        ing,mapped=load_files(conn,files,args)
        print('Creating EDA outputs ...',flush=True)
        summary=run_eda(conn,out,ing,mapped,args)
        print('Detecting snapshot clusters ...',flush=True)
        count,top=run_detection(conn,out,args)
        plot_candidates(top,out)
        summary['candidate_snapshot_count']=count
        (out/'eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        print(f'DONE: {count} candidate snapshot clusters; results in {out.resolve()}')
    finally: conn.close()

def main():
    parser=argparse.ArgumentParser(description='AIS Week 10: inspect, disk-backed EDA, snapshot DBSCAN clusters')
    sub=parser.add_subparsers(dest='cmd',required=True)
    for cmd in ['inspect','run']:
        p=sub.add_parser(cmd)
        p.add_argument('inputs',nargs='+',help='CSV/CSV.GZ files or directories')
        p.add_argument('--encoding',default='utf-8-sig')
        p.add_argument('--sep',default=',')
        for k in ALIASES: p.add_argument(f'--{k}-col',dest=k+'_col',help=f'Explicit {k} column')
        if cmd=='inspect':
            p.add_argument('--sample-rows',type=int,default=20)
            p.add_argument('--output',help='Write schema inspection JSON')
        else:
            p.add_argument('--out',default='outputs/week10')
            p.add_argument('--chunk-rows',type=int,default=50000)
            p.add_argument('--time-minutes',type=int,default=5)
            p.add_argument('--grid-deg',type=float,default=.01)
            p.add_argument('--eps-nm',type=float,default=.5)
            p.add_argument('--min-vessels',type=int,default=3)
            p.add_argument('--max-diameter-nm',type=float,default=None)
            p.add_argument('--max-vessels-per-bin',type=int,default=50000)
            p.add_argument('--max-plot-candidates',type=int,default=30)
            p.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if args.cmd=='inspect':inspect(args.inputs,args)
    else:
        if not (args.time_minutes>0 and args.grid_deg>0 and args.eps_nm>0 and args.min_vessels>=3):parser.error('Invalid thresholds')
        run(args)
if __name__=='__main__':main()
