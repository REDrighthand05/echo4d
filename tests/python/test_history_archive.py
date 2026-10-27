"""File archive counterexamples; synthetic bytes, no camera/hardware execution."""
import hashlib
import json
from pathlib import Path
import pytest
from echo4d import history
from echo4d.model import DataError


def fixture(tmp_path):
    media=tmp_path/'clip.webm';media.write_bytes(b'not-a-decoded-video')
    capture={'version':history.CAPTURE_VERSION,'time_base':history.TIMEBASE,
             'created_utc':'2026-10-28T00:00:00Z','duration_s':2,
             'media':{'file':media.name,'bytes':media.stat().st_size,'mime_type':'video/webm'},
             'imu':{'calibrated':False,'actual_head_pose':'unknown','samples':[]},
             'auxiliary_sensor_observations':[],'geometry_status':'not_reconstructed'}
    cap=tmp_path/'capture.json';cap.write_bytes((json.dumps(capture,separators=(',',':'))+'\r\n').encode())
    aux={'version':history.AUX_VERSION,'sensor_id':'sensorA','kind':'scan',
         'time_base':'device_clock_unaligned','units':'mm',
         'spatial_calibration':'unknown','samples':[{'t_s':1,'values':[1,2]}]}
    sensor=tmp_path/'sensor.json';sensor.write_bytes((json.dumps(aux,indent=1)+'\r\n').encode())
    return cap,media,sensor,tmp_path/'out'


def test_archive_retains_exact_original_json_bytes_and_hashes(tmp_path):
    cap,media,sensor,out=fixture(tmp_path)
    result=history.import_history(cap,media,out,(sensor,))
    timeline=json.loads((out/'history.timeline.json').read_text())
    assert (out/'capture-source.json').read_bytes()==cap.read_bytes()
    assert (out/'sensors/sensorA.json').read_bytes()==sensor.read_bytes()
    assert timeline['capture_source']['sha256']==hashlib.sha256(cap.read_bytes()).hexdigest()
    assert timeline['auxiliary_streams'][0]['source_sha256']==history.file_sha256(out/'sensors/sensorA.json')
    assert result['media_sha256']==history.file_sha256(media)
    assert timeline['auxiliary_streams'][0]['clock_status']=='unaligned'
    assert result['reconstruction']=='not_attempted'


@pytest.mark.parametrize('which',['media','capture','sensor'])
def test_input_change_during_copy_rolls_back(tmp_path,monkeypatch,which):
    cap,media,sensor,out=fixture(tmp_path);copy=history.shutil.copyfile
    def mutate(source,dest):
        copied=copy(source,dest)
        target={'media':media,'capture':cap,'sensor':sensor}[which]
        raw=target.read_bytes();target.write_bytes(bytes([raw[0]^1])+raw[1:])
        return copied
    monkeypatch.setattr(history.shutil,'copyfile',mutate)
    with pytest.raises(DataError,match='changed'):
        history.import_history(cap,media,out,(sensor,))
    assert not out.exists()


def test_duplicate_sensor_case_does_not_overwrite_windows_archive(tmp_path):
    cap,media,sensor,out=fixture(tmp_path);second=tmp_path/'second.json'
    data=json.loads(sensor.read_text());data['sensor_id']='SENSORA';second.write_text(json.dumps(data))
    with pytest.raises(DataError,match='Repeated'):
        history.import_history(cap,media,out,(sensor,second))
    assert not out.exists()


def test_existing_archive_never_overwritten(tmp_path):
    cap,media,sensor,out=fixture(tmp_path);out.mkdir();(out/'sentinel').write_text('keep')
    with pytest.raises(DataError,match='already exists'):
        history.import_history(cap,media,out,(sensor,))
    assert (out/'sentinel').read_text()=='keep'
