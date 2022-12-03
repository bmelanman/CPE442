  LAB5DIR = ../Lab5/src/
  LAB6DIR = ../Lab6/src/

  all: clean lab5 lab6

  lab5:
	make pi -C $(LAB5DIR)

  lab6:
	make pi -C $(LAB6DIR)

  restore_labs:
	make all -C $(LAB5DIR) && make all -C $(LAB6DIR)

  clean:
	make clean -C $(LAB5DIR) && make clean -C $(LAB6DIR)

  fix: restore_labs
