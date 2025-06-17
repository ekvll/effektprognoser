: '
Iterates over all files in the ../ep/cli/ directory.
For each file:
	- Check if it is a regular file (not a directory).
	- Extract the filename without its path.
	- Prints the filename without its file extension (the part before the first dot).

Inputs:
	- Files located in ../ep/cli/

Outputs:
	- Echoes the basename of each file (no extension) to stdout.

Example:
	If ../ep/cli/ contains:
		foo.py
		bar.txt
		baz.sh
	The output will be:
		foo
		bar
		baz
'

echo "Available CLI commands:"
path=../ep/cli/
for file in $path*; do
	[ -f "$file" ] && basename="${file##*/}" && echo "${basename%%.*}"
done
