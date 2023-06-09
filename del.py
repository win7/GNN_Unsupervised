import sys

print('cmd entry:', sys.argv)

for item in sys.argv:
    print(item, type(item))