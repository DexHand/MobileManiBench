from unimanip.utils.general_utils import *

# save GenieSimAssets
def save_room_geniesim():
    # process all GenieSimAssets rooms
    for room_name, value_ in SCENE_ANALYSIS['room']['GenieSimAssets'].items():
        # locate usd_path
        usd_path = osp.join(ASSET_DIR, 'room/GenieSimAssets/{}/scene.usda'.format(room_name))
        if room_name in ['scenes/iros/base_factory', 'scenes/iros/sofa_table']:
            usd_path = osp.join(ASSET_DIR, 'room/GenieSimAssets/{}_scene.usda'.format(room_name))
        # locate save_path
        save_path = usd_path.replace('/room/', '/room/scenes/')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        shutil.copyfile(usd_path, save_path)

# save GRScenes
def save_room_grscenes():
    # process all GRScenes rooms
    for room_name, value_ in SCENE_ANALYSIS['room']['GRScenes'].items():
        # locate usd_path
        usd_path = osp.join(ASSET_DIR, 'room/GRScenes/{}/scene.usd'.format(room_name))
        # locate save_path
        save_path = usd_path.replace('/room/', '/room/scenes/')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        shutil.copyfile(usd_path, save_path)


# prepare GenieSimAssets
def prepare_room_geniesim():
    # process all GenieSimAssets rooms
    for room_name, value_ in SCENE_ANALYSIS['room']['GenieSimAssets'].items():
        # locate usd_path
        usd_path = osp.join(ASSET_DIR, 'room/scenes/GenieSimAssets/{}/scene.usda'.format(room_name))
        if room_name in ['scenes/iros/base_factory', 'scenes/iros/sofa_table']:
            usd_path = osp.join(ASSET_DIR, 'room/scenes/GenieSimAssets/{}_scene.usda'.format(room_name))
        # locate save_path
        save_path = usd_path.replace('/room/scenes/', '/room/')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        shutil.copyfile(usd_path, save_path)


# prepare GRScenes
def prepare_room_grscenes():
    # process all GRScenes rooms
    for room_name, value_ in SCENE_ANALYSIS['room']['GRScenes'].items():
        # locate usd_path
        usd_path = osp.join(ASSET_DIR, 'room/scenes/GRScenes/{}/scene.usd'.format(room_name))
        # locate save_path
        save_path = usd_path.replace('/room/scenes/', '/room/')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        shutil.copyfile(usd_path, save_path)

# # save scene.usd
# save_room_geniesim()
# save_room_grscenes()

# # prepare scene.usd
# prepare_room_geniesim()
# prepare_room_grscenes()

