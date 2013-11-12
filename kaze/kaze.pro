TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += \
    ../src/kaze_match.cpp \
    ../src/kaze_features.cpp \
    ../src/kaze_compare.cpp \
    ../src/lib/utils.cpp \
    ../src/lib/nldiffusion_functions.cpp \
    ../src/lib/KAZE.cpp

OTHER_FILES += \
    ../CMakeLists.txt

HEADERS += \
    ../src/kaze_match.h \
    ../src/kaze_features.h \
    ../src/kaze_compare.h \
    ../src/lib/utils.h \
    ../src/lib/nldiffusion_functions.h \
    ../src/lib/KAZE.h \
    ../src/lib/config.h

