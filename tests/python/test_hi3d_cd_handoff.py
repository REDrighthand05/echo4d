"""C -> D real file-format handoff checks; inputs here are synthetic, NOT measured furniture."""
from pathlib import Path
import runpy
import sys
import json
import numpy as np
import pytest
pytest.importorskip('trimesh', reason='Dedicated C/D integration workflow installs optional mesh dependency')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from echo4d.holdout import read_echo_ply
from echo4d.model import DataError
from real_furniture_extract import extract, EvidenceError
from hi3d_furniture_compare import read_furniture_geometry

fixture=runpy.run_path(str(Path(__file__).with_name('test_real_furniture_extract.py')))['fixture']

def test_real_export_from_c_can_be_consumed_by_d_without_ply_schema_weakening(tmp_path):
    root, manifest, spec=fixture(tmp_path)
    report=extract(manifest,root,tmp_path/'out')
    assert report['verified_furniture'] is False
    vertices, faces=read_furniture_geometry(tmp_path/'out/observed-object-world.npz')
    assert len(vertices)==4 and len(faces)==2
    assert np.allclose(np.sort(vertices,axis=0), np.sort(np.asarray([
        [-.5,-.5,2],[.5,-.5,2],[.5,.5,2],[-.5,.5,2]
    ]),axis=0))
    assert np.array_equal(np.load(tmp_path/'out/selected-source-face-indices.npy'),[0,1])
    # The preview PLY deliberately stays geometry-only, and MUST NOT be
    # silently accepted by the core's strict measured-TSDF PLY reader.
    with pytest.raises(DataError):
        read_echo_ply(tmp_path/'out/observed-object-world.ply')

def test_repeated_renamed_capture_is_rejected_before_export(tmp_path):
    root, manifest, spec=fixture(tmp_path)
    spec['frames'][1]=dict(spec['frames'][0],name='different_name')
    manifest.write_text(json.dumps(spec))
    with pytest.raises(EvidenceError,match='Repeated source'):
        extract(manifest,root,tmp_path/'out')
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('field', ['extra','bad_face','nonfinite','object_array'])
def test_geometry_npz_d_reader_rejects_malformed_or_unsafe_files(tmp_path,field):
    path=tmp_path/'mesh.npz'
    v=np.asarray([[0,0,0],[1,0,0],[0,1,0]],float)
    f=np.asarray([[0,1,2]],np.int64)
    if field=='extra':
        np.savez(path,vertices=v,faces=f,claim_of_provenance=np.array([1]))
    elif field=='bad_face':
        np.savez(path,vertices=v,faces=np.asarray([[0,1,4]],np.int64))
    elif field=='nonfinite':
        v[0,0]=np.nan
        np.savez(path,vertices=v,faces=f)
    else:
        np.savez(path,vertices=np.array([{'untrusted':True}],dtype=object),faces=f)
    with pytest.raises(DataError):
        read_furniture_geometry(path)


def test_observed_only_cli_preserves_report_but_cannot_signal_geometry_acceptance(tmp_path, monkeypatch):
    import hi3d_furniture_compare as d
    report = {'candidate_is_not_worse_on_frozen_metrics': True,
              'conclusion': 'OBSERVED_METRICS_PASS_GEOMETRY_INCONCLUSIVE'}
    monkeypatch.setattr(d, 'evaluate', lambda *args: report)
    out = tmp_path / 'report.json'
    code = d.main(['--ledger', 'unused', '--proof', 'unused',
                   '--root', str(tmp_path), '--out', str(out)])
    assert code == 3
    assert json.loads(out.read_text()) == report


def test_cd_resource_rejection_occurs_before_materialization(tmp_path, monkeypatch):
    import io
    import zipfile
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        'shape': (2_000_001, 3), 'fortran_order': False, 'descr': '<f8'})
    p = tmp_path / 'oversized.npz'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('vertices.npy', header.getvalue())
        z.writestr('faces.npy', b'not materialized')
    def forbidden(*args, **kwargs):
        pytest.fail('unbounded array load')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(DataError):
        read_furniture_geometry(p)
