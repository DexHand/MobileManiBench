from unimanip.utils.general_utils import *
from unimanip.utils.ycb_model import YCBObjModel
from unimanip.utils.partnet_model import PartNetObjModel
from unimanip.utils.unidoor_model import UniDoorObjModel


class RoomModel:
    def __init__(self, room_id, room_split, obj_model: PartNetObjModel | UniDoorObjModel | YCBObjModel, device='cuda:0'):
        """
        Room Model
        
        Parameters
        ----------
        room_id: int, room id
        obj_model: object model
        device: str, torch.Device, device for torch tensors
        """
        
        # init basic
        self.device = device
        # init object id, cat, type, size
        self.obj_id = obj_model.obj_id
        self.obj_cat = obj_model.obj_cat
        self.obj_type = obj_model.obj_type
        self.obj_size = obj_model.obj_size_rotated * obj_model.obj_scale
        # init room_id and room_infos
        self.room_num = 20
        self.room_id = room_id
        self.room_split = room_split
        self.read_room_infos()
    
    # read room_infos with room_id from SCENE_ANALYSIS
    def read_room_infos(self):
        # init room_place
        self.room_place = SCENE_ANALYSIS['object'][self.obj_type][self.obj_cat]['place']
        # init room_infos
        self.room_list = []
        # read all room_infos
        for room_group, room_names in SCENE_ANALYSIS['room'].items():
            for room_name, room_list in room_names.items():
                for room_infos in room_list:
                    if self.room_place in room_infos['types'] or self.obj_cat in room_infos['types']:
                        room_infos['group'], room_infos['name'], room_infos['place'] = room_group, room_name, self.room_place
                        if self.room_split == room_infos['split']: self.room_list.append(room_infos)
        # # append room_list to room_num
        # for room in random.sample(self.room_list, max(0, self.room_num - len(self.room_list))):
        #     self.room_list.append(room)
        # locate room_infos with room_id
        self.room_infos = self.room_list[self.room_id]
    
    # process obj_room_infos
    def process_obj_room_infos(self):
        # process room pose
        if self.room_place in ['space', 'outdoor']:
            pass
        elif self.room_place in ['wall', 'door']:
            self.room_infos['translation'][1] += self.obj_size[1]
        elif self.room_place in ['tabletop']:
            pass
        # process room usd_path
        if self.room_infos['group'] == 'Isaac':
            self.room_infos['usd_path'] = osp.join(ASSET_DIR, 'room/Isaac/{}.usd'.format(self.room_infos['name']))
        elif self.room_infos['group'] == 'GRScenes':
            self.room_infos['usd_path'] = osp.join(ASSET_DIR, 'room/GRScenes/{}/scene.usd'.format(self.room_infos['name']))
        elif self.room_infos['group'] == 'GenieSimAssets':
            self.room_infos['usd_path'] = osp.join(ASSET_DIR, 'room/GenieSimAssets/{}/scene.usda'.format(self.room_infos['name']))
            if self.room_infos['name'] in ['scenes/iros/base_factory', 'scenes/iros/sofa_table']:
                self.room_infos['usd_path'] = osp.join(ASSET_DIR, 'room/GenieSimAssets/{}_scene.usda'.format(self.room_infos['name']))
        return self.room_infos



if __name__ == '__main__':
    pass