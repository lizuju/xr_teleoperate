"""Color identity for the O6 palm cameras and the replacement right Dex1 camera."""

IMAGE_COLOR_PROFILE = "r1_wrist_device_rgb_v1"

# Recorded source versions that applied the O6 channel swap to the right Dex1 camera.
LEGACY_DEX1_SWAP_HASHES = {
    "57b490ca1138423c6aaf0ee294562f160e9e704633991e701354ef5284f27360",
    "a69c0bd5d66d7009809b3ede5c01dffd7abfedec01b75a6f38b4f35d38117903",
    "4e13cb675de21c11e3677f2b82723a4376dd177b7bd3db0d8d1c508ff2296420",
    "81c756d017aab0239ab05304648d418e8f63b891e3c08119069246ba6565b287",
}


def recorded_red_blue_swaps(info):
    profile = info.get("image_color_profile")
    if profile == IMAGE_COLOR_PROFILE:
        return ()
    if profile is not None:
        raise ValueError(f"Unknown recorded image_color_profile: {profile}")
    if (info.get("end_effectors") or {}).get("right") != "dex1_1":
        return ()
    digest = (info.get("code_sha256") or {}).get("teleop/teleop_hand_and_arm.py")
    if digest not in LEGACY_DEX1_SWAP_HASHES:
        raise ValueError("Unverified Dex1 recording color: inspect the source before exporting or reviewing")
    return ("color_3",)
