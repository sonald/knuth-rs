Read a UTF-8 text file with line numbers.

- Paths may be absolute or relative to the process working directory.
- Reads support a 1-based `offset` and line `limit`.
- Maximum returned content per call is 32 KiB (32768 bytes). Larger requests fail without returning partial content.
