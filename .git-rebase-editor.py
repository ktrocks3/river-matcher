import sys

p = sys.argv[1]
lines = open(p).read().splitlines()

n = 0
out = []

for line in lines:
    if line.startswith("pick "):
        n += 1
        out.append(line if (n - 1) % 100 == 0 else line.replace("pick ", "fixup ", 1))
    else:
        out.append(line)

open(p, "w").write("\n".join(out) + "\n")