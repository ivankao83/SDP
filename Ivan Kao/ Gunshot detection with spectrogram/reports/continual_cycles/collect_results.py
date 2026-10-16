"""Compile saved evaluations; no model parameters or thresholds are selected here."""
import json
from pathlib import Path
ROOT=Path.cwd()
OUT=Path('reports/continual_cycles');OUT.mkdir(exist_ok=True)

def read(path):return json.loads(Path(path).read_text())

def old_metrics(path):
 r=read(path);matrix=r['confusion_matrix_rows_true_columns_predicted']
 tn,fp=matrix[0];fn,tp=matrix[1]
 return {'recordings':tn+fp+fn+tp,'gunshots_detected':tp,'gunshots_missed':fn,'false_alarms':fp,'correct_non_gunshots':tn,
         'accuracy':r['accuracy'],'gunshot_recall':r['gunshot']['recall'],'gunshot_precision':r['gunshot']['precision'],
         'gunshot_f1':r['gunshot']['f1-score'],'roc_auc':r['roc_auc'],'threshold':r['threshold'],
         'false_positive_rate':fp/(tn+fp),'negative_rejection_rate':tn/(tn+fp)}

cycles=[]
for folder,name,number in [('run5_sesa_valid','SESA mixed sounds',5),('run6_fsdd','FSDD spoken digits',6),
                           ('run7_edge_distilled','EDGE gunshots',7),('run8_urban','UrbanSound8K subset',8)]:
 path=Path('runs')/folder/'cycle_results.json'
 if not path.exists():continue
 r=read(path)
 note='Selected trained weights.' if r['selected_epoch'] else 'Prior weights retained; threshold recalibrated.'
 cycles.append({'name':name,'dataset':r['dataset'],'run':f'Run #{number}',
                'epoch':r['selected_epoch'],'before':r['before']['test'],'after':r['after']['test'],
                'all_before':r['before']['all'],'all_after':r['after']['all'],
                'note':note,'report':str(path),'checkpoint_sha256':r['checkpoint_sha256']})
bg=read('runs/run4_background_revision/background_recheck/test_metrics.json')['results'][0]
oldbg=read('runs/run4_initial_file_test/test_metrics.json')['results'][0]
history={'madBefore':old_metrics('reports/run2/baseline_mad.json'),
         'madAfter':old_metrics('reports/run2/test_mad.json'),
         'original':old_metrics('reports/run2/run1_original.json'),
         'originalAfter':old_metrics('reports/run2/test_original.json'),
         'backgroundBefore':oldbg,'backgroundAfter':bg,
         'backgroundThresholdBefore':oldbg['threshold'],'backgroundThresholdAfter':bg['threshold']}
result={'cycles':cycles,'history':history,'metric_unit':'source file / recording',
        'interpretation':'Primary comparisons use held-out files never used for optimization or threshold selection. Datasets are selected adaptively across cycles; this is development evidence, not an independent final certification.',
        'targets':{'mixed_accuracy':.9,'gunshot_recall':.85,'false_positive_rate':.05}}
latest=read('runs/run8_urban/final_new_datasets.json')['results']
result['latest']=[{'name':r['dataset'],'metrics':r['metrics']} for r in latest]
for label,folder in [('Original','final_original'),('MAD','final_mad')]:
 result['latest'].append({'name':label,'metrics':old_metrics(f'runs/run8_urban/{folder}/test_metrics.json')})
for label,path in [('ShotSpotter','runs/run8_urban/final_shotspotter/test_metrics.json'),('C3GD','runs/run8_urban/eval_c3gd/test_metrics.json')]:
 r=read(path)
 if 'results' in r:r=r['results'][0]
 result['latest'].append({'name':label,'metrics':{'recordings':r['recordings'],'accuracy':None,
     'gunshot_recall':r['recording_gunshot_recall'],'gunshot_precision':None,'gunshot_f1':None,'roc_auc':None,
     'false_positive_rate':None,'negative_rejection_rate':None,'threshold':r['threshold'],
     'gunshots_detected':r['detected'],'gunshots_missed':r['missed'],'false_alarms':0,'correct_non_gunshots':0}})
result['latestBackground']=read('runs/run8_urban/final_background/test_metrics.json')['results'][0]
(OUT/'workbook_data.json').write_text(json.dumps(result,indent=2))
lines=['Continued gunshot detection experiments','',result['interpretation'],'']
for r in cycles:
 lines += [r['run']+' '+r['name'],r['note']]
 for phase in ['before','after']:
  v=r[phase]
  fmt=lambda key:'n.a.' if v[key] is None else f'{v[key]:.2%}'
  lines.append(f"  {phase}: {v['recordings']} held-out files; accuracy {fmt('accuracy')}; recall {fmt('gunshot_recall')}; false-positive rate {fmt('false_positive_rate')}; TP={v['gunshots_detected']}, FN={v['gunshots_missed']}, FP={v['false_alarms']}, TN={v['correct_non_gunshots']}; threshold={v['threshold']:.2f}")
 lines.append('')
lines+=['Latest checkpoint across all held-out datasets (threshold 0.31)']
for row in result['latest']:
 v=row['metrics'];fmt=lambda key:'n.a.' if v[key] is None else f'{v[key]:.2%}'
 lines.append(f"  {row['name']}: N={v['recordings']}; accuracy {fmt('accuracy')}; recall {fmt('gunshot_recall')}; false-positive rate {fmt('false_positive_rate')}")
lines+=['','Only the SESA cycle adopted trained weight changes. FSDD, EDGE, and UrbanSound8K training candidates did not improve the validation selection objective, so inherited weights were retained with validation-selected thresholds.',
        'The UrbanSound8K first test before any training on that corpus already met the working mixed-data targets (96.36% accuracy, 90.91% recall, 2.27% false-positive rate).',
        'Performance is not uniformly good: EDGE and ShotSpotter gunshot recall remain weak. More cycles did not establish universal accuracy.',
        'Known-background recheck includes train/validation files and must not be reported as unseen-test performance.']
(OUT/'RESULTS.txt').write_text('\n'.join(lines))
print('\n'.join(lines))
