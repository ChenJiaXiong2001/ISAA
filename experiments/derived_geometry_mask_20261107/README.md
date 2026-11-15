# Mask-aware derived geometry screen

This isolated experiment guards the hand/face derived distance and direction
channels when torso anchors are incomplete. It does not alter the production
`FacePairFusion` model or any existing entry point.

`geometry_only` keeps torso-relative coordinates and masks cross-hand derived
geometry. `center_scale` additionally uses center zero and scale one when all
four torso anchors are not available. Both variants are scratch-only for new
training; loading a formal checkpoint is inference-only exploratory evidence.
