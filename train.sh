#!/bin/bash
# export JAVA_HOME="" # Set here if not set in environment
# export CUDA_PATH="" # Set here if not set in environment
JCUDA_VERSION="0.7.0a" # Fix if needed
MEMSIZE="-Xmx14G"
export JAVA_OPTS="${MEMSIZE} -Xms128M -Dfile.encoding=UTF-8" # Set as much memory as possible
BIDMACH_ROOT="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_ROOT=`readlink -f "${BIDMACH_ROOT}"`
else 
  while [ -L "${BIDMACH_ROOT}" ]; do
    BIDMACH_ROOT=`readlink "${BIDMACH_ROOT}"`
  done
fi
BIDMACH_ROOT=`dirname "$BIDMACH_ROOT"`
pushd "${BIDMACH_ROOT}"  > /dev/null
BIDMACH_ROOT=`pwd`
BIDMACH_ROOT="$( echo ${BIDMACH_ROOT} | sed s+/cygdrive/c+c:+ )" 
JCUDA_LIBDIR="${BIDMACH_ROOT}/lib"
LIBDIR="${BIDMACH_ROOT}/lib"
if [ -e java_native_path.txt ]; then
  JAVA_NATIVE=`cat java_native_path.txt`
else 
  JAVA_NATIVE=`java getnativepath`
  echo ${JAVA_NATIVE} > java_native_path.txt
fi
if [ `uname` = "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${LIBDIR}:/usr/local/cuda/lib:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIBDIR}:${LIBDIR}/cuda:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" 
fi
export JAVA_NATIVE=${JAVA_NATIVE}:${LD_LIBRARY_PATH}:${DYLD_LIBRARY_PATH}:
popd > /dev/null
if [ "$OS" = "Windows_NT" ]; then
    if [ ! "${JAVA_HOME}" = "" ]; then
        JAVA_HOME=`${BIDMACH_ROOT}/shortpath.bat "${JAVA_HOME}"`
	    export JAVA_HOME=`echo ${JAVA_HOME} | sed 's_\\\\_/_g'`/bin
    fi
fi

BIDMACH_LIBS="${LIBDIR}/BIDMat.jar;${LIBDIR}/ptplot.jar;${LIBDIR}/ptplotapplication.jar;${LIBDIR}/jhdf5.jar;${LIBDIR}/commons-math3-3.2.jar;${LIBDIR}/lz4-1.3.jar"

JCUDA_LIBS="${JCUDA_LIBDIR}/jcuda-${JCUDA_VERSION}.jar;${JCUDA_LIBDIR}/jcublas-${JCUDA_VERSION}.jar;${JCUDA_LIBDIR}/jcufft-${JCUDA_VERSION}.jar;${JCUDA_LIBDIR}/jcurand-${JCUDA_VERSION}.jar;${JCUDA_LIBDIR}/jcusparse-${JCUDA_VERSION}.jar"

ALL_LIBS=";${LIBDIR}/IScala.jar;${BIDMACH_ROOT}/BIDMach.jar;${BIDMACH_LIBS};${JCUDA_LIBS};${JAVA_HOME}/lib/tools.jar"

if [ "$OS" = "Windows_NT" ]; then
    if [ ! "${CUDA_PATH}" = "" ]; then
	    NEWPATH=`${BIDMACH_ROOT}/shortpath.bat "${CUDA_PATH}"`
	    NEWPATH=`echo ${NEWPATH} | sed 's_\\\\_/_g'`/bin
    fi
    DJAVA_NATIVE="-Djava.library.path=${LIBDIR};${NEWPATH}"
else
    ALL_LIBS=`echo "${ALL_LIBS}" | sed 's/;/:/g'`
    DJAVA_NATIVE="-Djava.library.path=${JAVA_NATIVE}"
fi
if [ ! `uname` = "Darwin" ]; then
    export JAVA_OPTS="${DJAVA_NATIVE} ${JAVA_OPTS}"
fi

pio train -- --driver-memory 16g --executor-memory 8g --conf spark.driver.maxResultSize=3g --conf spark.network.timeout=500s --conf spark.akka.frameSize=2047
