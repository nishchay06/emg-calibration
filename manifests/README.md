# Test-user manifests

These files list the exact HDF5 archive members required by the official
`user0` through `user7` configs at upstream commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

| User | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `user0` | 10 | 2 | 2 | 14 |
| `user1` | 9 | 2 | 2 | 13 |
| `user2` | 9 | 2 | 2 | 13 |
| `user3` | 8 | 2 | 2 | 12 |
| `user4` | 8 | 2 | 2 | 12 |
| `user5` | 8 | 2 | 2 | 12 |
| `user6` | 8 | 2 | 2 | 12 |
| `user7` | 8 | 2 | 2 | 12 |
| **Total** | **68** | **16** | **16** | **100** |

Regenerate the files from a pinned upstream checkout:

```bash
./scripts/generate_test_user_manifests.py /path/to/emg2qwerty
```

Verify that the checked-in files are current without modifying them:

```bash
./scripts/generate_test_user_manifests.py /path/to/emg2qwerty --check
```

`test-users-sessions.txt` is the sorted union used for staging all eight users.
`test-users.json` records the pinned commit and per-split counts. These manifests
contain filenames only; no participant recordings or checkpoint data are stored
in this repository.
