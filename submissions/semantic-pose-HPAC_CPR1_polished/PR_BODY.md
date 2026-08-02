# submission name:

semantic-pose-HPAC_CPR1_polished

# upload zipped `archive.zip`

Archive URL: `<replace with the public curl -L archive.zip URL>`

SHA-256: `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`

Size: `186,724` bytes

# report.txt

```
=== Evaluation results over 600 samples ===
Average PoseNet Distortion 0.00000688
Average SegNet Distortion 0.00029639
Submission file size 186,724 bytes
Original uncompressed 37,545,489
Rate 0.00497327
score ... = 0.16
```

# does your submission require gpu for evaluation (inflation)?

Yes. The decoder targets the Linux NVIDIA T4 evaluation rail and requires CUDA
for inflation.

# did you include the compression script? and want it to be merged?

Yes. `compress.sh` retrieves the promoted F26 archive from `ARCHIVE_URL`,
verifies its exact hash and ZIP layout, and writes `archive.zip` without
packaging the exploratory training pipeline.

# is this submission competitive or innovative? explain why

Competitive candidate. The promoted F26 archive uses the F24S representation,
a fixed-boundary int6 residual table, RC64 token decoding, and bounded frame-0
carrier compensation. The
reproducibility run recorded a `0.16226842` score over all 600 public pairs;
the challenge workflow remains the authoritative evaluation.

# additional comments

The decoder was refactored after the earlier CPR1 review feedback. It keeps
only the components used by F26 and removes the unused floating HPAC and range
decoder implementations. `verify_submission.py` pins the promoted archive by
hash and checks its wire format before evaluation.
