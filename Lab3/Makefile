  # Compiler for C++
  CC = g++

  # Package Manager for opencv4
  PKG_MNG = `pkg-config --cflags --libs opencv4`

  # Compiler flags:
  #  -g    			adds debugging information to the executable file
  #  -Wall 			turns on most, but not all, compiler warnings
  #  -std=c++11 	implements all of the major features of the C++11 standard
  CFLAGS  = $(PKG_MNG) -std=c++11 -g -Wall

  # Target build executable:
  TARGET = main

  default: $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

  clean:
	$(RM) $(TARGET)