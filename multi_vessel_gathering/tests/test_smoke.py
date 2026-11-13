"""Run with: python -m unittest discover -s events/multi_vessel_gathering/tests -v"""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mvg import detect_bucket, candidate_indices, select_columns
from types import SimpleNamespace
class TestDetection(unittest.TestCase):
    def test_schema_aliases(self):
        x=select_columns(['MMSI','lat','lon','posDt','sog'],{})
        self.assertEqual(x['time'],'posDt')
    def test_actual_vendor_schema(self):
        cols=['source_id','mmsi','longitude','latitude','sog','record_time','fetched_at']
        x=select_columns(cols,{})
        self.assertEqual(x,{'vessel':'mmsi','lat':'latitude','lon':'longitude','time':'record_time','speed':'sog'})
    def test_3_ship_cluster(self):
        args=SimpleNamespace(min_vessels=3,eps_nm=.5,max_diameter_nm=None)
        # tuple format: vessel, lat, lon, sog, gx, gy
        points=[('1',25.,121.,1,12100,2500),('2',25.0004,121.0002,1,12100,2500),('3',25.0003,121.0004,1,12100,2500),('4',26.,122.,2,12200,2600)]
        groups=detect_bucket(points,args)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['count'],3)
    def test_insufficient_ships(self):
        self.assertEqual(candidate_indices([('1',25.,121.,1,12100,2500)],3),[])
if __name__=='__main__': unittest.main()
