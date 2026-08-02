# semantic-pose-HPAC_CPR1_polished

F26 is an `F24S` archive with a fixed-boundary int6 residual table, native RC64
token decoding, and a stored frame-0 selector. The promoted archive is 186,724 bytes and has SHA-256
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.

`archive.zip` and `report.txt` are intentionally ignored by the challenge
repository. Before opening a pull request, upload the exact archive to a URL
that supports `curl -L`, then paste that URL into the pull-request body.

## Validation

With the promoted archive present in this directory, run:

```bash
python verify_submission.py
```

The verifier checks the archive hash, the single `p` payload, and the fixed
F26 wire format. Inflation requires CUDA. `inflate.sh` compiles the small RC64
decoder into a temporary directory, performs no network access, and writes
`0.raw` through the required challenge interface.

## Rebuild the submitted archive

```bash
bash compress.sh
```

`compress.sh` downloads the promoted archive from `ARCHIVE_URL`, verifies its
byte size, SHA-256, and ZIP layout, then writes `archive.zip`. Use
`ARCHIVE_URL=<published-archive-url> bash compress.sh`, `--archive-url` to
provide a mirror, or `--out` to select a different output path. It
intentionally reuses the frozen promoted artifact instead of packaging the
exploratory CUDA search pipeline.

## Decoder scope

This submission contains only the runtime decoder required by F26. It omits
the unused floating-point HPAC implementation and legacy range decoder from
the earlier CPR1 work. The remaining integer HPAC model, renderer, and F24S
parser are the components exercised by the promoted archive.
