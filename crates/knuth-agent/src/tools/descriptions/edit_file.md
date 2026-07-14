Edit a text file by exact string replacement.

- Paths may be absolute or relative to the process working directory.
- `old_string` must be non-empty and different from `new_string`.
- By default, the match must be unique. Set `replace_all=true` to replace every match.
- Common text encodings are detected and the file is written back using its original encoding.
