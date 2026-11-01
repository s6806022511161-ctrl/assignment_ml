"""Reproducible orchid experiments. Run --help before starting."""
import argparse
import csv
import hashlib
import html
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASSES = ['cattleya', 'dendrobium', 'oncidium', 'phalaenopsis', 'vanda']
ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_html_report(out, benchmark):
    def fmt(value, digits=2):
        return f'{value:.{digits}f}'
    cards = []
    rows = []
    for result in benchmark:
        name = result['model']
        rows.append(
            '<tr>'
            f'<td>{html.escape(name)}</td>'
            f'<td>{fmt(result["best_val_accuracy_pct"])}%</td>'
            f'<td>{fmt(result["accuracy_pct"])}%</td>'
            f'<td>{fmt(result["macro_precision"] * 100)}%</td>'
            f'<td>{fmt(result["macro_recall"] * 100)}%</td>'
            f'<td>{fmt(result["macro_f1"] * 100)}%</td>'
            f'<td>{fmt(result["parameters_m"])} M</td>'
            f'<td>{fmt(result["latency_median_ms"])} ms</td>'
            '</tr>'
        )
        cards.append(f'''<section class="model">
          <h2>{html.escape(name)}</h2>
          <div class="figures">
            <figure><img src="{name}/learning_curves.png"><figcaption>Learning curves</figcaption></figure>
            <figure><img src="{name}/confusion_matrix.png"><figcaption>Confusion matrix</figcaption></figure>
          </div>
          <p><a href="{name}/per_class.csv">Per-class metrics</a> ·
             <a href="{name}/predictions.csv">Predictions</a> ·
             <a href="{name}/history.csv">Training history</a></p>
        </section>''')
    counts = []
    with (out / 'dataset_counts.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            counts.append(f'<tr><td>{html.escape(row["label"])}</td><td>{row["train"]}</td><td>{row["val"]}</td><td>{row["test"]}</td></tr>')
    page = f'''<!doctype html><html lang="th"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Thai Orchid Classification - Experiment Report</title>
    <style>
    body{{font-family:Segoe UI,Tahoma,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#24202a;background:#f7f4f8}}
    h1,h2{{color:#6a1b9a}} .box,.model{{background:white;padding:22px;margin:18px 0;border-radius:14px;box-shadow:0 3px 14px #0001}}
    table{{border-collapse:collapse;width:100%;overflow:auto;display:block}} th,td{{padding:10px 13px;border:1px solid #ddd;white-space:nowrap}} th{{background:#6a1b9a;color:white}}
    .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}} img{{width:100%;height:auto}} figcaption{{text-align:center;color:#555}}
    a{{color:#6a1b9a}} code{{background:#eee;padding:2px 5px}} .note{{color:#555}}
    </style></head><body>
    <h1>Thai Orchid Classification - Experiment Report</h1>
    <p>ResNet-50 vs EfficientNet-B0 vs ViT-B/16</p>
    <section class="box"><h2>Dataset</h2><table><thead><tr><th>Class</th><th>Train</th><th>Validation</th><th>Test</th></tr></thead><tbody>{''.join(counts)}</tbody></table></section>
    <section class="box"><h2>Model benchmark</h2><table><thead><tr><th>Model</th><th>Best Val Accuracy</th><th>Test Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1</th><th>Parameters</th><th>Median latency*</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <p class="note">* Model-only inference, batch size 1; excludes image loading and preprocessing. Hardware: {html.escape(str(benchmark[0]['device']))}.</p>
    <img src="benchmark.png" alt="Model benchmark"></section>
    {''.join(cards)}
    <section class="box"><h2>Files for slides</h2><p><a href="benchmark.csv">benchmark.csv</a> · <a href="dataset_counts.csv">dataset_counts.csv</a> · <a href="config.json">config.json</a></p></section>
    </body></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')


def scan(folder):
    records = []
    for path in sorted(folder.rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        label = path.relative_to(folder).parts[0].lower()
        label = 'cattleya' if label == 'cattariya' else label
        if label not in CLASSES:
            raise ValueError(f'Unknown class: {path}')
        with Image.open(path) as image:
            image = image.convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        records.append(dict(path=str(path.resolve()), label=label, sha256=digest))
    return records


def prepare(args, out):
    train, val, test = [scan(args.data / s) for s in ['train', 'val', 'test']]
    if not train or not test:
        raise ValueError('Train and test must contain labelled images.')
    hashes = {}
    duplicates = []
    clean = {}
    for split, rows in [('train', train), ('val', val), ('test', test)]:
        clean[split] = []
        for row in rows:
            key = row['sha256']
            if key in hashes:
                previous = hashes[key]
                duplicates.append(dict(path=row['path'], previous=previous['path'], split=split))
                if previous['split'] != split or previous['label'] != row['label']:
                    save_csv(out / 'duplicate_conflicts.csv', duplicates)
                    raise ValueError('Identical pixels across splits or conflicting labels. See duplicate_conflicts.csv.')
                continue
            hashes[key] = dict(row, split=split)
            clean[split].append(row)
    train, val, test = [clean[s] for s in ['train', 'val', 'test']]
    if not val:
        grouped = defaultdict(list)
        for row in train:
            grouped[row['label']].append(row)
        train = []
        for label in CLASSES:
            rows = grouped[label]
            if len(rows) < 2:
                raise ValueError(f'Not enough training images: {label}')
            random.Random(args.seed).shuffle(rows)
            n = max(1, min(len(rows)-1, round(len(rows)*args.val_fraction)))
            val.extend(rows[:n])
            train.extend(rows[n:])
    split_rows = {'train': train, 'val': val, 'test': test}
    for split, rows in split_rows.items():
        if set(r['label'] for r in rows) != set(CLASSES):
            raise ValueError(f'{split} must include all five classes.')
    manifest = [dict(row, split=s) for s, rows in split_rows.items() for row in rows]
    save_csv(out / 'split_manifest.csv', manifest)
    save_csv(out / 'duplicates_removed.csv', duplicates)
    counts = [dict(label=c, **{s: sum(r['label']==c for r in rows) for s, rows in split_rows.items()}) for c in CLASSES]
    save_csv(out / 'dataset_counts.csv', counts)
    print('Dataset:', {s:len(rows) for s,rows in split_rows.items()}, flush=True)
    return split_rows


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, CLASSES.index(row['label'])


def build(name, head_only):
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if head_only:
            model.requires_grad_(False)
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if head_only:
            model.requires_grad_(False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    else:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=5)
        if head_only:
            model.requires_grad_(False)
            model.head.requires_grad_(True)
    return model


def evaluate(model, loader, device):
    truth, prediction, probabilities = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            p = model(x.to(device)).softmax(1).cpu()
            truth.extend(y.tolist())
            prediction.extend(p.argmax(1).tolist())
            probabilities.extend(p.tolist())
    return truth, prediction, probabilities


def experiment(args):
    if args.output.exists():
        raise ValueError('Output already exists; choose a new --output to preserve previous results.')
    args.output.mkdir(parents=True)
    out = args.output
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='auto' else ('cpu' if args.device=='auto' else args.device))
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    save_json(out/'config.json', dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, classes=CLASSES, actual_device=str(device), torch_version=torch.__version__))
    splits = prepare(args,out)
    if args.prepare_only:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
    norm = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), norm])
    train_tf = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(.2,.2,.2), transforms.ToTensor(),norm])
    benchmark = []
    for name in args.models:
        torch.manual_seed(args.seed)
        loaders = {s:DataLoader(Images(rows,train_tf if s=='train' else tf),batch_size=args.batch_size,shuffle=s=='train',num_workers=args.workers,pin_memory=device.type=='cuda') for s,rows in splits.items()}
        folder = out/name
        folder.mkdir()
        model = build(name,args.head_only).to(device)
        lr = args.lr if args.lr is not None else (1e-4 if name.startswith('vit') else 3e-4)
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
        criterion = nn.CrossEntropyLoss()
        best, stale, history = -1,0,[]
        started = time.perf_counter()
        for epoch in range(1,args.epochs+1):
            row = dict(epoch=epoch,lr=optimizer.param_groups[0]['lr'])
            for phase in ['train','val']:
                model.train(phase=='train' and not args.head_only)
                total=correct=loss_sum=0
                for batch,(x,y) in enumerate(loaders[phase],1):
                    x,y=x.to(device),y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(phase=='train'):
                        logits=model(x)
                        loss=criterion(logits,y)
                        if phase=='train':
                            loss.backward()
                            optimizer.step()
                    total+=len(y)
                    correct+=(logits.argmax(1)==y).sum().item()
                    loss_sum+=loss.item()*len(y)
                    if batch%20==0:
                        print(f'{name} epoch {epoch} {phase} {batch}/{len(loaders[phase])}',flush=True)
                row[phase+'_loss']=loss_sum/total
                row[phase+'_accuracy']=correct/total
            scheduler.step()
            history.append(row)
            save_csv(folder/'history.csv',history)
            if row['val_accuracy']>best:
                best,stale=row['val_accuracy'],0
                torch.save(model.state_dict(),folder/(name+'_best.pth'))
                save_json(folder/'best_epoch.json',row)
            else:
                stale+=1
            print(row,flush=True)
            if args.patience and stale>=args.patience:
                break
        training_seconds=time.perf_counter()-started
        model.load_state_dict(torch.load(folder/(name+'_best.pth'),map_location=device,weights_only=True))
        y,p,prob=evaluate(model,loaders['test'],device)
        precision,recall,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
        report=classification_report(y,p,target_names=CLASSES,output_dict=True,zero_division=0)
        save_json(folder/'classification_report.json',report)
        save_csv(folder/'per_class.csv',[dict(label=c,**report[c]) for c in CLASSES])
        save_csv(folder/'predictions.csv',[dict(path=r['path'],actual=CLASSES[a],predicted=CLASSES[b],**{c:float(v) for c,v in zip(CLASSES,pr)}) for r,a,b,pr in zip(splits['test'],y,p,prob)])
        cm=confusion_matrix(y,p,labels=range(5))
        save_csv(folder/'confusion_matrix.csv',[dict(actual=c,**dict(zip(CLASSES,row.tolist()))) for c,row in zip(CLASSES,cm)])
        fig,ax=plt.subplots(figsize=(8,7))
        ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,cmap='Blues',xticks_rotation=45,colorbar=False)
        ax.set_title(name+' - Test set')
        fig.tight_layout(); fig.savefig(folder/'confusion_matrix.png',dpi=200); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,metric in zip(axes,['loss','accuracy']):
            for phase in ['train','val']:
                ax.plot([r['epoch'] for r in history],[r[phase+'_'+metric] for r in history],label=phase)
            ax.set(xlabel='Epoch',ylabel=metric,title=name); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=200); plt.close(fig)
        sample=next(iter(loaders['test']))[0][:1].to(device)
        def sync():
            if device.type=='cuda': torch.cuda.synchronize()
        times=[]
        with torch.inference_mode():
            for _ in range(10): model(sample)
            sync()
            for _ in range(args.latency_repeats):
                sync(); t=time.perf_counter(); model(sample); sync()
                times.append((time.perf_counter()-t)*1000)
        benchmark.append(dict(model=name,accuracy_pct=100*accuracy_score(y,p),macro_precision=precision,macro_recall=recall,macro_f1=f1,weighted_f1=report['weighted avg']['f1-score'],parameters_m=sum(v.numel() for v in model.parameters())/1e6,latency_median_ms=float(np.median(times)),latency_p95_ms=float(np.percentile(times,95)),training_seconds=training_seconds,best_val_accuracy_pct=best*100,test_images=len(y),device=str(device)))
        save_csv(out/'benchmark.csv',benchmark)
        latest = benchmark[-1]
        print(
            f'TEST {name}: accuracy={latest["accuracy_pct"]:.2f}% | '
            f'macro_precision={latest["macro_precision"]*100:.2f}% | '
            f'macro_recall={latest["macro_recall"]*100:.2f}% | '
            f'macro_f1={latest["macro_f1"]*100:.2f}% | '
            f'latency={latest["latency_median_ms"]:.2f} ms/image',
            flush=True,
        )
        del model,optimizer
        if device.type=='cuda': torch.cuda.empty_cache()
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,title in zip(axes,['accuracy_pct','parameters_m','latency_median_ms'],['Test Accuracy (%)','Parameters (M)','Model-only latency (ms), batch 1']):
        ax.bar([r['model'] for r in benchmark],[r[key] for r in benchmark]); ax.set_title(title); ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(out/'benchmark.png',dpi=200); plt.close(fig)
    make_html_report(out, benchmark)
    print('\nFINAL BENCHMARK', flush=True)
    for result in benchmark:
        print(
            f'{result["model"]:24s} | test accuracy {result["accuracy_pct"]:6.2f}% | '
            f'macro F1 {result["macro_f1"]*100:6.2f}% | '
            f'{result["parameters_m"]:6.2f} M params | '
            f'{result["latency_median_ms"]:8.2f} ms/image',
            flush=True,
        )
    print('Complete:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data'/'dataset')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument(
        '--models',
        nargs='+',
        choices=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224'],
        default=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224'],
        help='models to train (default: all three models)',
    )
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float)
    p.add_argument('--patience',type=int,default=5,help='0 disables early stopping')
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--latency-repeats',type=int,default=30)
    p.add_argument('--head-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not 0<args.val_fraction<1 or min(args.epochs,args.batch_size,args.threads,args.latency_repeats)<1 or args.patience<0 or args.workers<0:
        p.error('Invalid numeric arguments')
    experiment(args)


if __name__=='__main__':
    main()
