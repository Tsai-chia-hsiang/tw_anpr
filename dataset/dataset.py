import os
from tqdm import tqdm
import gc
from typing import Literal
from pathlib import Path
from datetime import datetime, timedelta
import re
import numpy as np
import cv2
import json 

def read_json(jf:os.PathLike):
    with open(jf, "r") as jfp:
        return json.load(jfp)

def date_from_ref_root(ref_root:Path) -> tuple[str,  datetime]:
    """
    Returns:
    --
    Tuple(station name, date)
    """
    print(ref_root)
    vname = re.split(r'[_,-,T]+', ref_root.name)
    t = f"{vname[0]}-{vname[1]}:{vname[2]}:{vname[3]}"
    return datetime.strptime(t,  "%Y-%m-%d-%H:%M:%S")

def get_veh_and_lp_path_from_dir(d:Path, ext:Literal['png', 'jpg']='png') -> tuple[Path, Path|None]:
    """
    Returns the path to the vehicle crop and optionally the license plate crop
    from the specified directory.

    Args:
    -----
    - d (Path):
        The directory containing the vehicle and (optionally) license plate image crops.
        It must contain the image 'veh.{ext}' where {ext} is the specified extension.
        The function will return the path to this image as `d/'veh.{ext}'`.

        If a license plate image 'lp.{ext}' exists in the directory, its path will also be returned.

    - ext (str, optional):
        The file extension of the image files. Default is 'png'. The value must be either 'png' or 'jpg'.

    Returns:
    --------
    - tuple of Paths:
        If the license plate crop exists, both the vehicle and license plate image paths are returned as a tuple.\\
        Otherwise, (veh_path. ```None```)
    """
    assert d.is_dir()
    veh = d/f"veh.{ext}"
    assert veh.is_file()
    lp = d/f"lp.{ext}"
    if not lp.is_file():
        return veh, None
    
    return veh, lp

class VehicleTracklet():

    def __init__(self, ref_root:Path, tid:int, ext:Literal['png', 'jpg']='png', fps=5, label:dict=None) -> None:
       
        self.ref_root:Path = ref_root
        self.img_ext = ext
        self.fps = fps
        self.tid:int = tid
        self.station = self.ref_root.parent.parent.name      
        self.lp_label = label
        self.veh_crops = sorted([_ for _ in ref_root.glob(f"*.{ext}")],key=lambda x: int(x.stem))
        self.start_date = date_from_ref_root(ref_root=self.ref_root.parent) + timedelta(seconds=int(self.veh_crops[0].stem) // self.fps)
        self.end_time = self.start_date + timedelta(seconds=int(self.veh_crops[-1].stem)//self.fps)
        self._len_ = len(self.veh_crops)

    def __repr__(self) -> str:
        return f"vehicle {self.tid} from {self.station} frame:{self.veh_crops[0].stem}-{self.veh_crops[-1].stem}({self.start_date}-{self.end_time})"
    
    def __len__(self) -> int:
        return self._len_
    
    def __getitem__(self, idx:int) -> tuple[np.ndarray, Path, dict|None, datetime]:
        
        frame_id = self.veh_crops[idx].stem
        
        label = self.lp_label.get(frame_id, None)
        
        timestamp = self.start_date + timedelta(seconds=int(frame_id)//self.fps)
        
        veh_crop = cv2.imread(str(self.veh_crops[idx]))

        return veh_crop, self.veh_crops[idx], label, timestamp
        

class Station():
    
    def __init__(self, station_name:str, data_root:os.PathLike, img_ext:Literal['png', 'jpg']='png') -> None:
        
        self.sname = station_name
        self.img_ext = img_ext
        self.root = Path(data_root)/self.sname
        assert self.root.is_dir()
        self.veh_tracklets= self._parsing_tracklets_from_videos()
        self._videos_name = tuple(self.veh_tracklets.keys())
        
    def _parsing_tracklets_from_videos(self) -> dict[str, dict[str, None]]:
        ret:dict[str, dict[str, None]] = {}
        for vname in self.root.iterdir():
            if vname.is_dir():
                ret[vname.name] = {
                    tid.name:None 
                    for tid in vname.iterdir() if tid.is_dir()
                }
                ret[vname.name]['need_parsing'] = True
        return ret
    
    @property
    def video_names(self)->tuple:
        return self._videos_name
    
    def get_videos(self, name: str, paring_bar:bool=False):
        
        if name in self.veh_tracklets:
            if 'need_parsing' in self.veh_tracklets[name]:
                if paring_bar:
                    p = tqdm(self.veh_tracklets[name], total=len(self.veh_tracklets[name]))
                else:
                    p = self.veh_tracklets[name]
                
                for tid in p:
                    meta_data = read_json(self.root/name/"meta_data.json")
                    global_label = {}
                    if (self.root/name/"lp_label.json").is_file():
                        global_label = read_json(self.root/name/"lp_label.json")
        
                    if tid.isdigit():
                        self.veh_tracklets[name][tid] = VehicleTracklet(
                            ref_root=self.root/name/tid,
                            tid=tid, ext=self.img_ext,
                            fps=meta_data['fps'],
                            label=global_label.get(tid, None)
                        )
                
                del self.veh_tracklets[name]['need_parsing']
                gc.collect()
            
            return self.veh_tracklets[name]
        
        return f"No {name} such a video under {self.root}"

    def __repr__(self) -> str:
        return f"{self.root} : {self._videos_name}"
    

if __name__ == "__main__":
    s = Station(station_name='新南所', data_root=Path("tracklet_lp"))
    print(s, s.video_names)
    v = s.get_videos(s.video_names[0])
    print(v)