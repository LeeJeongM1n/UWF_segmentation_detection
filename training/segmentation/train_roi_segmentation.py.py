#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()  

from pathlib import Path

import torch
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import segmentation_models_pytorch as smp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ============================================================
# Dataset & Joint Transform 
# ============================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_fnames, mask_fnames, transform):
        self.image_paths = [os.path.join(image_dir, f) for f in image_fnames]
        self.mask_paths  = [os.path.join(mask_dir, f) for f in mask_fnames]
        self.transform   = transform
        self.image_fnames= image_fnames
        self.mask_fnames = mask_fnames

        # for datacheck (image-mask pair)
        for idx, (img_f, mask_f) in enumerate(zip(image_fnames, mask_fnames)):
            ib, _ = os.path.splitext(img_f)
            mb, _ = os.path.splitext(mask_f)
            if mb.endswith("_mask"):
                mb = mb[:-5]
            if ib != mb:
                print(f"[WARNING] idx={idx}: image='{img_f}' vs mask='{mask_f}' (base mismatch)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")
        # print(np.unique(np.array(mask)))
        image, mask = self.transform(image, mask)
        img_name = self.image_fnames[idx]
        return {"image": image, "mask": mask, "name": img_name}


def create_dataloader(image_dir, mask_dir, image_filenames, mask_filenames, batch_size, transform, shuffle):
    ds = SegmentationDataset(image_dir, mask_dir, image_filenames, mask_filenames, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


class JointTransform:
    """
    - Resize to (H, W)
    - Optional h/v flip, color jitter
    - ToTensor + Normalize (Imagenet or custom)
    - Binary mask thresholding at 0.9
    """
    def __init__(self, input_size, do_flip=True, do_color_jitter=True, do_normalize=True, mean_std=None):
        if isinstance(input_size, int):
            resize_size = (input_size, input_size)
        elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            resize_size = tuple(input_size)
        else:
            raise ValueError("--input_size should be int or tuple/list of length 2")

        self.do_flip = do_flip
        self.do_color_jitter = do_color_jitter
        self.do_normalize = do_normalize

        self.resize_image = transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC)
        self.resize_mask  = transforms.Resize(resize_size, interpolation=Image.NEAREST)

        self.to_tensor = transforms.ToTensor()

        self.normalize = None
        if self.do_normalize:
            if mean_std is None:
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            else:
                m, s = mean_std
                self.normalize = transforms.Normalize(mean=m, std=s)

        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.5) if self.do_color_jitter else None #revised_1021

    def __call__(self, image, mask):
        image = self.resize_image(image)
        mask  = self.resize_mask(mask)

        if self.do_flip:
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                mask  = transforms.functional.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.vflip(image)
                mask  = transforms.functional.vflip(mask)

        # if self.jitter is not None:
        #     image = self.jitter(image)
        # 3. Image-Only Photometric Augmentations
        if self.jitter is not None: 
            # Apply Brightness and Contrast Jitter
            image = self.jitter(image)
            
            # Manual Gamma Correction (Requested range 0.5 to 2.0)
            gamma_factor = random.uniform(0.5, 2.0) #revised_1021
            image = transforms.functional.adjust_gamma(image, gamma_factor)

        image = self.to_tensor(image)
        mask  = self.to_tensor(mask)
        mask  = (mask >= 0.5).float()  # binary, threshold : 127.5
        ## UWF_Quality Dataset 학습 시 mask < 0.5로 설정: 
        ## (mask>=0.5).float()로 설정 시 convert('L') 에 의해서 학습에 사용되는 마스크의 노란부분(외곽)이 양성 값으로 변환됨 -> 원하는 값과 반대
        ## mask : RGBA type -> convert('L) : background >= 150 , eye <= 90

        if self.do_normalize:
            image = self.normalize(image)

        return image, mask


def denorm_img(img_tensor, rgb_mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    """ IMAGENET mean/std로 정규화한 이미지 복원 (C,H,W) -> (H,W,C) [0,1] """
    img = img_tensor.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"denorm_img expects [C,H,W], got {tuple(img.shape)}")
    C, H, W = img.shape
    if C != 3:
        raise ValueError(f"Unsupported channel count: {C}")
    mean, std = rgb_mean_std
    out = img.clone()
    for c in range(3):
        out[c] = out[c] * std[c] + mean[c]
    out = out.clamp(0, 1).numpy().transpose(1, 2, 0)
    return out


# ============================================================
# LightningModule
# ============================================================
class SmpModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        encoder_depth,
        in_channels,
        out_classes,
        seg_dropout,             
        encoder_weights="imagenet",
        bce_weight=0.5,
        base_lr=1e-4,
        weight_decay=1e-4,
        step_size=5,
        gamma=0.1,
        aux_params=None,     
        metrics_csv_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # 모델이 '확률'을 바로 반환
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=out_classes,
            activation="sigmoid",          # 확률 출력
            encoder_weights=encoder_weights,
            decoder_aspp_dropout=seg_dropout,
            aux_params=aux_params
        )
        self.bce_weight  = float(bce_weight)
        self.bce_fn      = nn.BCELoss()   # 확률 기준 BCE
        self.dice_fn     = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)

        # Optimizer/Scheduler 파라미터
        self._base_lr      = float(base_lr)
        self._weight_decay = float(weight_decay)
        self._step_size    = int(step_size)
        self._gamma        = float(gamma)

        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs  = []

        # 에폭별 CSV 작성을 위해 누적
        self.metrics_csv_path = metrics_csv_path
        self._row_cache = None 
        self._metric_cols = [
            "epoch",
            "train_loss","train_bce_loss","train_dice_loss","train_dice_score", "train_soft_dice",
            "valid_loss","valid_bce_loss","valid_dice_loss","valid_dice_score", "valid_soft_dice",
            "test_loss","test_bce_loss","test_dice_loss","test_dice_score", "test_soft_dice"
        ]

    def forward(self, image):
        out = self.model(image)   # 확률 반환
        if isinstance(out, (tuple, list)):
            prob = out[0]
        else:
            prob = out
        return prob  # [B,1,H,W] in [0,1]

    @staticmethod
    def _stats_from_outputs(outputs):
        if len(outputs) == 0:
            zero = torch.tensor(0.0)
            return dict(loss=zero, bce_loss=zero, dice_loss=zero, dice_score=zero, soft_dice=zero)
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        per_img_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        eps = 1e-7
        probs = torch.cat([x["prob"] for x in outputs]).double()
        gts   = torch.cat([x["mask"] for x in outputs]).double()
        inter_s = (probs * gts).sum().item()
        sum_s   = ((probs ** 2).sum() + (gts ** 2).sum()).item()
        soft_dice = float((2.0 * inter_s + eps) / (sum_s + eps))

        mean_dice_loss = torch.stack([x["dice_loss"] for x in outputs]).mean()
        mean_bce_loss  = torch.stack([x["bce_loss"]  for x in outputs]).mean()
        mean_loss      = torch.stack([x["loss"]      for x in outputs]).mean()
        
        return dict(loss=mean_loss, bce_loss=mean_bce_loss, dice_loss=mean_dice_loss, dice_score=per_img_dice, soft_dice=soft_dice)

    def _shared_step(self, batch):
        image, mask = batch if not isinstance(batch, dict) else (batch["image"], batch["mask"])
        assert image.ndim == 4 and mask.ndim == 4
        prob_mask   = self.forward(image)                    # 확률
        prob_mask   = prob_mask.clamp(1e-7, 1-1e-7)

        with torch.cuda.amp.autocast(enabled=False):
            p32, m32 = prob_mask.float(), mask.float()
            bce  = self.bce_fn(p32, m32)                    # 확률 기준 BCE
            dice = self.dice_fn(p32, m32)                   # 확률 기준 Dice
        loss = self.bce_weight * bce + (1.0 - self.bce_weight) * dice

        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {"dice_loss": dice, "bce_loss": bce, "loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "prob": prob_mask.detach().float().cpu(), "mask": mask.detach().float().cpu()}

    # ---------------- train ----------------
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.train_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        stats = self._stats_from_outputs(self.train_step_outputs)
        self.train_step_outputs.clear()
        self.log("train_loss", stats["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self._row_cache = {
        "epoch": int(self.current_epoch),
        "train_loss":       stats["loss"].item(),
        "train_bce_loss":   stats["bce_loss"].item(),
        "train_dice_loss":  stats["dice_loss"].item(),
        "train_dice_score": stats["dice_score"].item(),
        "train_soft_dice":  stats["soft_dice"],
        }


    # ---------------- validation (valid + test 동시) ----------------
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._shared_step(batch)
        if dataloader_idx == 0:
            self.valid_step_outputs.append(out)
        else:
            self.test_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        v_stats = self._stats_from_outputs(self.valid_step_outputs)
        self.valid_step_outputs.clear()        
        self.log("valid_loss", v_stats["loss"], prog_bar=True, on_step=False, on_epoch=True)

        t_stats = self._stats_from_outputs(self.test_step_outputs)
        self.test_step_outputs.clear()

        cur_ep = int(self.current_epoch)
        row = {}
        if self._row_cache is not None:
            if self._row_cache.get("epoch") in (cur_ep, cur_ep - 1):
                row.update(self._row_cache)

            self._row_cache = None
        
        row.setdefault("epoch", cur_ep)
        row.update({
            "valid_loss":       v_stats["loss"].item(),
            "valid_bce_loss":   v_stats["bce_loss"].item(),
            "valid_dice_loss":  v_stats["dice_loss"].item(),
            "valid_dice_score": v_stats["dice_score"].item(),
            "valid_soft_dice":  v_stats["soft_dice"],
            "test_loss":        t_stats["loss"].item(),
            "test_bce_loss":    t_stats["bce_loss"].item(),
            "test_dice_loss":   t_stats["dice_loss"].item(),
            "test_dice_score":  t_stats["dice_score"].item(),
            "test_soft_dice":   t_stats["soft_dice"],
        })
            

        if self.metrics_csv_path is not None:
            # 컬럼 순서 고정
            one = {k: row.get(k, None) for k in self._metric_cols}
            df1 = pd.DataFrame([one], columns=self._metric_cols)
            file_exists = os.path.exists(self.metrics_csv_path)
            df1.to_csv(self.metrics_csv_path, mode='a', index=False, header=not file_exists)


    # ---------------- optim (AdamW + StepLR) ----------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._base_lr, weight_decay=self._weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self._step_size, gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--base_folder', default='/mnt/richul_FM/UWF_Quality', type=str)
    # parser.add_argument('--mask_folder',  default='./images/mask_RPE',     type=str)
    parser.add_argument('--train_csv', default='train.csv', type=str)
    parser.add_argument('--valid_csv', default='valid.csv', type=str)
    parser.add_argument('--test_csv',  default='test.csv',  type=str)

    # model
    parser.add_argument('--arch_name',       default='DeepLabV3Plus', type=str)
    parser.add_argument('--encoder_name',    default='resnet50',      type=str)
    parser.add_argument('--encoder_depth',   default=5, type=int)
    parser.add_argument('--in_channels',     default=3, type=int)
    parser.add_argument('--out_classes',     default=1, type=int)
    parser.add_argument('--seg_dropout',     default=0.3, type=float)        # 기록용
    parser.add_argument('--bce_weight',      default=0.5, type=float)
    parser.add_argument('--encoder_weights', default='imagenet', type=str)

    # optim (AdamW 1e-4, wd 1e-4, StepLR(5, 0.1))
    parser.add_argument('--base_lr',    default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--step_size',  default=2, type=int)
    parser.add_argument('--gamma',      default=0.5, type=float)

    # run
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--input_size', nargs='+', default=[320, 320], type=int)
    parser.add_argument('--epochs',     default=30, type=int)
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--gpu_select', default="0", type=str)
    parser.add_argument('--exp_name',   default="", type=str)
    parser.add_argument('--augmented',  action='store_true')
    args = parser.parse_args()

    # 환경
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_select
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # GPU 사용 시 학습 속도가 느려지더라도 일관성을 보장하는 설정
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False

    set_seed(42)


    # ---- 통합 실험 폴더 구성: {output_dir}/{arch}-{encoder}[__{exp_name}]
    name_tag = f"{args.arch_name}-{args.encoder_name}"
    exp_name = args.exp_name 
    if exp_name:
        name_tag += f"__{exp_name}"

    exp_dir  = os.path.join(args.output_dir, name_tag)
    logs_dir = os.path.join(exp_dir, "logs")
    imgs_dir = os.path.join(exp_dir, "images/test")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)

    # ---- 데이터셋/로더
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    test_df  = pd.read_csv(args.test_csv)

    def lists_from_df(df: pd.DataFrame):
        img_names  = df["img_fname"].tolist()
        mask_names = df["mask_fname"].tolist()
        return img_names, mask_names

    train_fnames, train_mask_fnames = lists_from_df(train_df)
    valid_fnames, valid_mask_fnames = lists_from_df(valid_df)
    test_fnames,  test_mask_fnames  = lists_from_df(test_df)

    root_dir = Path(args.base_folder)
    # train_img_dir  = root_dir / "train" / "wfp_img"
    # train_mask_dir = root_dir / "train" / "wfp_mask"
    # valid_img_dir  = root_dir / "valid" / "wfp_img"
    # valid_mask_dir = root_dir / "valid" / "wfp_mask"
    # test_img_dir   = root_dir / "test"  / "wfp_img"
    # test_mask_dir  = root_dir / "test"  / "wfp_mask"

    train_img_dir  = root_dir / "train" / "YOLO_img"
    train_mask_dir = root_dir / "train" / "YOLO_mask"
    valid_img_dir  = root_dir / "valid" / "YOLO_img"
    valid_mask_dir = root_dir / "valid" / "YOLO_mask"
    test_img_dir   = root_dir / "test"  / "YOLO_img"
    test_mask_dir  = root_dir / "test"  / "YOLO_mask"

    mean_std = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    resize_size = args.input_size[0] if len(args.input_size) == 1 else tuple(args.input_size)
    train_tf = JointTransform(resize_size, do_flip=True,  do_color_jitter=True,  do_normalize=True, mean_std=mean_std)
    eval_tf  = JointTransform(resize_size, do_flip=False, do_color_jitter=False, do_normalize=True, mean_std=mean_std)

    train_loader = create_dataloader(str(train_img_dir), str(train_mask_dir), train_fnames, train_mask_fnames, args.batch_size, train_tf, shuffle=True)
    print("train drop_last: ", train_loader.drop_last)
    valid_loader = create_dataloader(str(valid_img_dir), str(valid_mask_dir),  valid_fnames, valid_mask_fnames, args.batch_size, eval_tf,  shuffle=False)
    print("valid drop_last: ", valid_loader.drop_last)
    test_loader  = create_dataloader(str(test_img_dir), str(test_mask_dir),  test_fnames,  test_mask_fnames,  args.batch_size, eval_tf,  shuffle=True)
    print("test drop_last: ", test_loader.drop_last)

    metrics_csv_path = os.path.join(logs_dir, "metrics.csv")

    # aux_params=dict( #revised_1021
    # dropout=0.1,               # dropout ratio, default is None
    # activation='sigmoid',      # activation function, default is None
    # classes=1,                 # define number of output labels
    # )

    # ---- 모델/콜백/트레이너 
    model = SmpModel(
        arch=args.arch_name,
        encoder_name=args.encoder_name,
        encoder_depth=args.encoder_depth,
        #aux_params=aux_params,
        in_channels=args.in_channels,
        out_classes=args.out_classes,
        seg_dropout=args.seg_dropout,            
        encoder_weights=args.encoder_weights,
        bce_weight=args.bce_weight,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        metrics_csv_path=metrics_csv_path, 
    ).to(device)
    #print(summary(model, input_size=(4, 3, 768, 768)))  
    print("###########################")
    first_conv = list(model.model.encoder.children())[0].weight
    print("Mean:", first_conv.mean().item(), "Std:", first_conv.std().item()) #revised_1021 if mean near 0 and std >0.1 --> imagenet pretrained
    print("###########################")

    ckpt_cb = ModelCheckpoint(
        dirpath=exp_dir,
        filename=f"{args.arch_name}-{args.encoder_name}-epoch={{epoch:02d}}-valid_loss={{valid_loss:.4f}}",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        callbacks=[ckpt_cb],
        precision=16,
        accumulate_grad_batches=4,
        logger=False,  
    )

    # validation 루프에 valid & test를 동시에 넣어 매 에폭마다 test도 평가
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[valid_loader, test_loader]
    )
    # --- validaion and test metrics ---
    best_ckpt_path = ckpt_cb.best_model_path

    model = SmpModel.load_from_checkpoint( #revised_1021
    checkpoint_path=best_ckpt_path, 
    map_location=device)

    # model = SmpModel.load_from_checkpoint(checkpoint_path = ckpt_cb.best_model_path, arch = args.arch_name, 
    #                                       encoder_name = args.encoder_name, encoder_depth = 5,  #aux_params=aux_params, 
    #                                       in_channels = 3, out_classes = 1, activation = "sigmoid", encoder_weights="imagenet" ,
    #                                       bce_weight = args.bce_weight, map_location=device) # seg_dropout = args.seg_dropout, 
    model.to(device)

    # ---- metrics.csv 
    if os.path.exists(metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)
    if not df.empty and "valid_loss" in df.columns:
        best_idx = int(df["valid_loss"].idxmin())
        best_epoch = df.loc[best_idx, "epoch"]
        best_row = df.loc[best_idx].copy()
        best_row["epoch"] = f"Best_epoch_{int(best_epoch)}"
        pd.DataFrame([best_row], columns=df.columns).to_csv(
            metrics_csv_path, mode='a', index=False, header=False
        )


    print("Best model saved to:", ckpt_cb.best_model_path)
    print("Experiment dir     :", exp_dir)
    print("Metrics CSV        :", metrics_csv_path)

    # ---- hparams.yaml 저장 
    try:
        import yaml
        hparams = {
            "arch_name": args.arch_name,
            "encoder_name": args.encoder_name,
            "seg_dropout": args.seg_dropout,
            "bce_weight": args.bce_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "input_size": args.input_size,
            "exp_name": exp_name,
        }
        with open(os.path.join(logs_dir, "hparams.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(hparams, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        # PyYAML이 없을 경우를 대비한 fallback (JSON)
        import json
        with open(os.path.join(logs_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(hparams, f, ensure_ascii=False, indent=2)

    # ---- 시각화 샘플 저장
    model.eval()
    bins = np.linspace(0.0, 1.0, 51)
    bin_thre = 0.5
    eps = 1e-7
    global_idx = 0

    from tqdm import tqdm

    with torch.inference_mode():
        for b_idx, batch in tqdm(enumerate(test_loader)): # total_loader #revised_1021
            images = batch["image"].to(device)
            gt_masks = batch["mask"]
            names = batch["name"]
            
            probs  = model(images)  # [B,1,H,W]
            if isinstance(probs, (tuple, list)):
                probs = probs[0]
            pr_probs = probs.squeeze(1)

            for i in range(images.size(0)):
                img_fname = names[i]
                img_name, _ = os.path.splitext(img_fname)
                image  = images[i].detach().cpu()
                img_np = denorm_img(image)
                gt_np  = gt_masks[i].detach().cpu().numpy().squeeze().astype(np.uint8)
                prob_np= pr_probs[i].detach().cpu().numpy().squeeze()
                pred_np= (prob_np > bin_thre).astype(np.uint8)

                # 전체 이미지 중 마스크 픽셀 비율 계산
                _, H, W = image.shape                             
                gt_pixels = int((gt_np > 0).sum())
                total_pixels = int(H * W) if (H > 0 and W > 0) else 1
                pixel_percent = 100.0 * gt_pixels / total_pixels

                inter_h = np.logical_and(pred_np == 1, gt_np == 1).sum()
                sum_h   = pred_np.sum() + gt_np.sum()
                hard_dice = (2.0 * inter_h + eps) / (sum_h + eps)

                inter_s = (prob_np * gt_np).sum()
                sum_s   = (prob_np**2).sum() + (gt_np**2).sum()
                soft_dice = (2.0 * inter_s + eps) / (sum_s + eps)

                gt_overlay   = np.ma.masked_where(gt_np == 0, gt_np)
                pred_overlay = np.ma.masked_where(pred_np == 0, pred_np)

                plt.figure(figsize=(16, 4))
                plt.subplot(1, 4, 1)
                plt.imshow(img_np)
                plt.title(f"Input\n{hard_dice:.4f}")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(img_np)
                plt.imshow(gt_overlay, cmap="Reds",  alpha=0.7); plt.title("Input+GT")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(img_np)
                plt.imshow(pred_overlay, cmap="Greens",alpha=0.7); plt.title("Input+Pred")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.hist(prob_np.ravel(), bins=bins, edgecolor="black")
                plt.axvline(bin_thre, linestyle="--")
                plt.xlim(prob_np.min(), prob_np.max())
                plt.xlabel("Pr(mask)"); plt.ylabel("# pixels")
                plt.title(f"SoftDice={soft_dice:.4f}")

                save_path = os.path.join(imgs_dir, f"px{pixel_percent:.2f}%_{soft_dice:.4f}_{img_name}.png")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
                global_idx += 1


if __name__ == "__main__":
    main()
