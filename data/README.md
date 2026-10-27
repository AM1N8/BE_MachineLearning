# Input data

Supply these competition files here, or pass their directory using `--data`:

```text
train.csv
test.csv
sample_submission.csv
```

Uppercase filenames such as `TRAIN.csv` are accepted. The directory must contain exactly one case-insensitive match for each filename.

- Training traces are headerless and start with a user trigram, then a browser.
- Test traces are headerless and start with the browser; there is no label field.
- The sample header is exactly `RowId,prediction`, with IDs in test row order.
- Traces have variable numbers of fields. The parser protects quoted fields and commas inside parentheses, configuration tags and chain tags.

The original inspected data has 3,279 training sessions, 324 test sessions and 247 users. Data is not duplicated into this folder. In the parent repository, use `--data ../data`.
