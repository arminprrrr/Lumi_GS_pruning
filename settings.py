from lfs_plugins.props import (
    PropertyGroup,
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    FloatVectorProperty,
)


class GuardSettings(PropertyGroup):
    enabled = BoolProperty(
        default=True,
        name="Enabled",
        description="Enable automatic Gaussian suppression during training",
    )

    warmup_iters = IntProperty(
        default=0,
        min=0,
        max=10000,
        step=10,
        name="Warmup iterations",
        description="Do not suppress before this iteration",
    )

    apply_every = IntProperty(
        default=1,
        min=1,
        max=100,
        step=1,
        name="Apply every N steps",
        description="1 = run every iteration boundary",
    )

    center_mode = EnumProperty(
        items=[
            ("auto_once", "Auto once", "Capture center once from current Gaussians"),
            ("manual", "Manual", "Use the manual center below"),
        ],
        name="Center mode",
        description="How the 3D crop center is chosen",
    )

    center = FloatVectorProperty(
        default=(0.0, 0.0, 0.0),
        size=3,
        min=-100.0,
        max=100.0,
        subtype="XYZ",
        name="Center XYZ",
        description="Manual 3D crop center in world space",
    )

    enable_radius = BoolProperty(
        default=False,
        name="Match outside radius",
        description="Match Gaussians farther than Radius from the center",
    )

    radius = FloatProperty(
        default=1.0,
        min=0.0,
        max=50.0,
        step=0.1,
        precision=3,
        name="Radius",
        description="World-space radius from center",
    )

    enable_max_axis = BoolProperty(
        default=True,
        name="Match oversized Gaussians",
        description="Match Gaussians whose largest axis exceeds Max axis scale",
    )

    max_axis = FloatProperty(
        default=0.17,
        min=0.0,
        max=10.0,
        step=0.01,
        precision=4,
        name="Max axis scale",
        description="Match when max(scale_x, scale_y, scale_z) exceeds this",
    )

    enable_aspect = BoolProperty(
        default=False,
        name="Match stretched Gaussians",
        description="Match Gaussians with extreme aspect ratio",
    )

    max_aspect = FloatProperty(
        default=10.0,
        min=1.0,
        max=1000.0,
        step=1.0,
        precision=2,
        name="Max aspect ratio",
        description="Match when max_axis and min_axis exceeds this",
    )

    fade_matched = BoolProperty(
        default=False,
        name="Fade matched",
        description="Set matched Gaussians to a target opacity",
    )

    opacity_target = FloatProperty(
        default=0.10,
        min=0.001,
        max=1.0,
        step=0.01,
        precision=3,
        name="Opacity target",
        description="Visible opacity target for matched Gaussians",
    )

    shrink_matched = BoolProperty(
        default=True,
        name="Shrink matched",
        description="Multiply only matched Gaussians by a scale factor",
    )

    scale_multiplier = FloatProperty(
        default=0.50,
        min=0.01,
        max=1.0,
        step=0.01,
        precision=3,
        name="Scale multiplier",
        description="0.50 halves matched Gaussian scale;",
    )

    log_each_hit = BoolProperty(
        default=True,
        name="Log updates",
        description="Write suppression activity to the LichtFeld log",
    )

    hard_delete = BoolProperty(
        default=False,
        name="Hard delete (disabled here)",
        description="Compatibility only; live mode fades instead of deleting.",
    )
