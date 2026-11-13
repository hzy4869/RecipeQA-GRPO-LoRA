
import sys,json,unittest,hashlib,datetime
from pathlib import Path
r=Path('/root/autodl-tmp/rl_recipeqa');h=r/'research/grounding_equivariance_20261106'
sys.path.insert(0,str(h));import orbit_diagnostic as o
import torch
assert not torch.cuda.is_initialized()
suite=unittest.TestLoader().discover(str(h),pattern='test_orbit_diagnostic.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==9 and not result.skipped
p=r/'data/manifests/val.jsonl';assert o.sha256_file(p)==o.DEV_SHA
orbit=o.make_orbit(o.read_jsonl(p));assert len(orbit)==384
from PIL import Image
files=sorted({str((r/'data/images'/p).resolve()) for x in orbit for p in x['image_paths_display']})
for f in files:
 assert Path(f).is_relative_to(r/'data/images')
 with Image.open(f) as im: im.verify()
assert len(files)==64 and not torch.cuda.is_initialized()
receipt={'status':'passed_actual_remote_CPU_orbit_preflight','tests_run':result.testsRun,'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),'cuda_initialized':torch.cuda.is_initialized(),'actual_records':384,'actual_recipes':16,'distinct_decoded_images':64,'source_manifest_sha256':o.sha256_file(p),'source_sha256':{n:o.sha256_file(h/n) for n in ['orbit_diagnostic.py','test_orbit_diagnostic.py']},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(h/'cpu_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
