# Demo artifact bundle

These files are the reviewed artifacts used by the public 10-class ResNet18
demo. They must be used together; mixing files from different training runs can
invalidate the feature layout, thresholds, and class mapping.

The bundle contains:

- frozen training reference features
- habit normality model
- trajectory projector and evidence nodes
- error-risk ranker
- multi-expert correction model

The runtime never reads truth labels from an uploaded image. Training labels
were used when producing these offline artifacts.
