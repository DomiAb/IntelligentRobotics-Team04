import pprint
import robomimic.utils.file_utils as FileUtils

ckpts = [
    "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/ckpt/dr_view_85.pth",
    "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/ckpt/d0_2views_100.pth",
    "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/ckpt/dr_2views_95.pth",
    "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/ckpt/tbd.pth"
]

for ckpt_path in ckpts:
    print("\n" + "=" * 80)
    print("CKPT:", ckpt_path)

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    print("all_obs_keys:")
    print(config.all_obs_keys)

    print("\nobservation.modalities:")
    pprint.pprint(config.observation.modalities)

    print("\nrollout horizon:")
    try:
        print(config.experiment.rollout.horizon)
    except Exception:
        print("N/A")

    # 1) 看 checkpoint 里有没有 shape metadata
    print("\nshape metadata in checkpoint:")
    for k in ["shape_metadata", "shape_meta", "observation_shape", "obs_shape"]:
        if k in ckpt_dict:
            print(f"\nFound key: {k}")
            pprint.pprint(ckpt_dict[k])
        else:
            print(f"{k}: not found")

    # 2) 尝试重点打印 obs 的 shape
    if "shape_metadata" in ckpt_dict:
        sm = ckpt_dict["shape_metadata"]
        print("\nDetailed shape_metadata:")
        pprint.pprint(sm)

        if "all_shapes" in sm:
            print("\nall_shapes:")
            for obs_key, shape in sm["all_shapes"].items():
                print(f"  {obs_key}: {shape}")

        if "all_obs_keys" in sm:
            print("\nshape_metadata all_obs_keys:")
            print(sm["all_obs_keys"])

    if "shape_meta" in ckpt_dict:
        sm = ckpt_dict["shape_meta"]
        print("\nDetailed shape_meta:")
        pprint.pprint(sm)

        if "all_shapes" in sm:
            print("\nall_shapes:")
            for obs_key, shape in sm["all_shapes"].items():
                print(f"  {obs_key}: {shape}")