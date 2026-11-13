"""Small real API/integration pilots, separate from every selected screen model."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/rl_recipeqa')
HERE=ROOT/'exploratory/reward_search_20261105'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'ops'))
from recipeqa.modeling import sha256_file,write_json
from wait_and_evaluate import require_gpu_idle

def main():
 os.chdir(ROOT)
 if ROOT.stat().st_dev==Path('/').stat().st_dev:raise RuntimeError('not_data_disk')
 if json.loads((HERE/'probe_state.json').read_text())['status']!='completed_development_probe':raise RuntimeError('probe_not_finished')
 locks=[]
 for name in ['formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline','reward_search_20261105']:
  h=(ROOT/('metadata/'+name+'.lock')).open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 require_gpu_idle()
 p=HERE/'pilot_state.json'
 if p.exists():raise RuntimeError('pilot_state_exists_no_restart')
 stream=ROOT/'data/manifests/train_visits.jsonl'
 if sha256_file(stream)!='e502e427cdfb0306e9ee8e581ad86ae159f0a7d96de1049716d81bb961622e20':raise RuntimeError('oldstream_changed')
 manifest=HERE/'pilot_prefix16.jsonl';manifest.write_bytes(b''.join(stream.read_bytes().splitlines(keepends=True)[:16]))
 source_sha={name:sha256_file(HERE/name) for name in ['train_reward_search.py','rewards.py','pilot_pipeline.py']}
 state={'status':'running','pid':os.getpid(),'source_sha256':source_sha,'commands':[],'test_access':False}
 write_json(p,state)
 for method,steps,save,batch in [('grpo',8,8,16),('sft',2,2,8)]:
  require_gpu_idle()
  output=HERE/('pilot_'+method)
  command=[sys.executable,str(HERE/'train_reward_search.py'),'--method',method,
   '--model-path',str(ROOT/'models/Qwen3-VL-2B-Instruct'),'--revision','89644892e4d85e24eaac8bacfd4f463576704203',
   '--image-root',str(ROOT/'data/images'),'--manifest',str(manifest),'--output-dir',str(output),
   '--initial-adapter',str(ROOT/'runs/sft_seed42/checkpoint-256'),'--max-steps',str(steps),'--save-steps',str(save),
   '--gradient-accumulation-steps',str(batch),'--seed','44']
  if method=='grpo':command+=['--generation-batch-size','16','--num-generations','8','--temperature','1.3','--reward','success_gate_pair']
  entry={'method':method,'arguments':command,'started_unix':time.time()};state['commands'].append(entry)
  try:
   with (HERE/('pilot_'+method+'.log')).open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    state.update(active_child_pid=child.pid,phase=method);write_json(p,state)
    entry['returncode']=child.wait(timeout=900)
   if entry['returncode']:raise RuntimeError(method+'_pilot_failed')
   entry['elapsed_seconds']=time.time()-entry['started_unix']
   outcome=json.loads((output/'train_outcome.json').read_text())
   if outcome['global_step']!=steps or not outcome['actual_visit_sequence_verified'] or not outcome['visual_weights_unchanged']:raise RuntimeError('invalid_pilot_outcome')
   entry['outcome']=outcome
   state.update(active_child_pid=None);write_json(p,state)
  except BaseException as error:
   if 'child' in locals() and child.poll() is None:
    child.terminate()
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:child.kill();child.wait()
   state.update(status='failed',error=str(error),active_child_pid=None);write_json(p,state);raise
 state.update(status='completed_real_integration_pilots',phase='grpo_and_sft_verified',active_child_pid=None)
 write_json(p,state)

if __name__=='__main__':main()
