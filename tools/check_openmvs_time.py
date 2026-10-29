"""Reproduce the pinned utility bug with glibc fortify, then check the backport."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from patch_openmvs_time import patch

source=(Path(sys.argv[1])/'libs/Common/Util.h').read_text()
prefix='#include <cstdint>\n#include <cstdio>\n#include <string>\n#include <cassert>\n#include <limits>\nusing String=std::string;\n#define _sntprintf snprintf\n'
main='''
int main(){
 assert(formatTime(0)=="0ms");
 assert(formatTime(1500)=="1s500ms");
 assert(formatTime(61001)=="1m1s1ms");
 assert(formatTime(61001,1)=="1m1s");
 for(unsigned a=0;a<5;++a){auto s=formatTime(std::numeric_limits<int64_t>::max(),a);assert(!s.empty()&&s.size()<128);}
 return 0;
}
'''
with tempfile.TemporaryDirectory() as directory:
    results={}
    for name,text in [('original',source),('fixed',patch(source))]:
        start=text.index('static String formatTime(')
        end=text.index('\n\tstatic String toString(',start)
        code=Path(directory)/(name+'.cpp');code.write_text(prefix+text[start:end]+main)
        executable=code.with_suffix('')
        subprocess.run(['g++','-O2','-D_FORTIFY_SOURCE=3','-fstack-protector-strong',str(code),'-o',str(executable)],check=True)
        result=subprocess.run([str(executable)],capture_output=True,text=True)
        results[name]={'returncode':result.returncode,'stderr':result.stderr.strip()}
    assert results['original']['returncode'] != 0 and 'buffer overflow' in results['original']['stderr']
    assert results['fixed']['returncode'] == 0
    print(json.dumps(results))
