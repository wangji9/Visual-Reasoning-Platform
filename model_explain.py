import torch
import cv2
import numpy as np

class ModelExplainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.hook_handles = []
        self.feature_maps = {}

    def register_hook(self, layer_name):
        # 先移除旧hook
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []
        self.feature_maps = {}
        # 注册新hook
        def hook_fn(module, input, output):
            # 兼容 tuple 输出
            if isinstance(output, tuple):
                for o in output:
                    if hasattr(o, 'detach'):
                        self.feature_maps[layer_name] = o.detach().cpu().numpy()
                        break
            elif hasattr(output, 'detach'):
                self.feature_maps[layer_name] = output.detach().cpu().numpy()
        # 支持字符串查找
        layer = dict([*self.model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)

    def get_all_layer_names(self):
        return [name for name, _ in self.model.named_modules() if name]

    def get_feature_map(self, img, layer_name):
        self.register_hook(layer_name)
        # 预处理
        if isinstance(img, np.ndarray):
            # resize到32的倍数（如640x640），与YOLOv8推理一致
            h, w = img.shape[:2]
            new_h = (h + 31) // 32 * 32
            new_w = (w + 31) // 32 * 32
            img_resized = cv2.resize(img, (new_w, new_h))
            x = img_resized.astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
        else:
            x = img
        x = x.to(self.device)
        with torch.no_grad():
            _ = self.model(x)
        fmap = self.feature_maps.get(layer_name)
        return fmap

    def featuremap_to_heatmap(self, fmap):
        # fmap: (1, C, H, W) or (C, H, W)
        if fmap is None:
            return None
        if fmap.ndim == 4:
            fmap = fmap[0]
        fmap = np.mean(fmap, axis=0)  # (H, W)
        fmap -= fmap.min()
        fmap /= (fmap.max() + 1e-8)
        fmap = (fmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
        return heatmap

    def featuremap_channel_to_heatmap(self, fmap, channel):
        # fmap: (1, C, H, W) or (C, H, W)
        if fmap is None:
            return None
        if fmap.ndim == 4:
            fmap = fmap[0]
        if channel < 0 or channel >= fmap.shape[0]:
            return None
        fmap_ch = fmap[channel]
        fmap_ch -= fmap_ch.min()
        fmap_ch /= (fmap_ch.max() + 1e-8)
        fmap_ch = (fmap_ch * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(fmap_ch, cv2.COLORMAP_JET)
        return heatmap

    def featuremap_all_channels_to_heatmaps(self, fmap):
        # 返回所有通道的热力图列表
        if fmap is None:
            return []
        if fmap.ndim == 4:
            fmap = fmap[0]
        heatmaps = []
        for ch in range(fmap.shape[0]):
            fmap_ch = fmap[ch]
            fmap_ch -= fmap_ch.min()
            fmap_ch /= (fmap_ch.max() + 1e-8)
            fmap_ch = (fmap_ch * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(fmap_ch, cv2.COLORMAP_JET)
            heatmaps.append(heatmap)
        return heatmaps
