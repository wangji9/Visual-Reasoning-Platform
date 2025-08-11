import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny(img, threshold1=100, threshold2=200):
    return cv2.Canny(img, threshold1, threshold2)

def invert(img):
    return cv2.bitwise_not(img)

def threshold(img, thresh=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return th

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# 新增图像处理函数

def median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def emboss(img):
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

def laplacian_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    edge = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# 1.1 图像翻转
def flip(img, mode=1):
    """
    mode: 0-垂直翻转, 1-水平翻转, -1-水平垂直翻转
    """
    return cv2.flip(img, mode)

# 1.2 幂运算和对数运算
def power_law(img, gamma=1.0):
    img_float = np.float32(img) / 255.0
    out = np.power(img_float, gamma)
    out = np.uint8(np.clip(out * 255, 0, 255))
    return out

def log_transform(img):
    img_float = np.float32(img) + 1.0
    out = np.log(img_float)
    out = out / np.max(out) * 255
    return np.uint8(out)

# 直方图均衡化
def hist_equalize(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# 图像增强（对比度拉伸）
def contrast_stretch(img):
    in_min = np.min(img)
    in_max = np.max(img)
    out = (img - in_min) * (255.0 / (in_max - in_min))
    return np.uint8(np.clip(out, 0, 255))

# 2.1 迭代法阈值分割
def iterative_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T = np.mean(gray)
    while True:
        G1 = gray[gray > T]
        G2 = gray[gray <= T]
        T_new = 0.5 * (np.mean(G1) + np.mean(G2))
        if abs(T - T_new) < 1:
            break
        T = T_new
    _, th = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return th

# 2.2 OTSU法
def otsu_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# 2.3 利用边缘改进阈值分割（边缘引导阈值）
def edge_guided_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mean_edge = np.mean(gray[edges > 0])
    _, th = cv2.threshold(gray, mean_edge, 255, cv2.THRESH_BINARY)
    return th

# 2.4 基于局部图像特征的可变阈值分割（自适应阈值）
def adaptive_threshold(img, blockSize=11, C=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, blockSize, C)
    return th

# 2.5 区域增长分割（简单实现）
def region_growing(img, seed=None, thresh=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = np.zeros_like(gray, np.uint8)
    if seed is None:
        seed = (h // 2, w // 2)
    stack = [seed]
    mask[seed] = 255
    seed_val = gray[seed]
    while stack:
        y, x = stack.pop()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                if abs(int(gray[ny, nx]) - int(seed_val)) < thresh:
                    mask[ny, nx] = 255
                    stack.append((ny, nx))
    return mask

# 3.1 提取目标边界
def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0,255,0), 2)
    return out, contours

# 3.2 计算目标的质心、长轴、短轴等参数
def contour_features(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour) if len(contour) >= 5 else ((0,0),(0,0),0)
    return {"centroid": (cx, cy), "major_axis": ma, "minor_axis": MA, "angle": angle}

# 3.3 计算边界线段的n阶统计矩
def contour_moments(contour, n=3):
    return cv2.moments(contour)

# 3.4 区域描绘子
def region_descriptors(contour, gray_img):
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    mean_val = cv2.mean(gray_img, mask=mask)[0]
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_img, mask=mask)
    return {
        "area": area,
        "perimeter": perimeter,
        "mean": mean_val,
        "min": min_val,
        "max": max_val
    }

# 3.5 基于灰度直方图的统计矩
def histogram_moments(gray_img, mask=None):
    # 保证输入为灰度图
    if len(gray_img.shape) == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
    hist = cv2.calcHist([gray], [0], mask, [256], [0,256]).flatten()
    hist = hist / np.sum(hist)
    mean = np.sum(hist * np.arange(256))
    var = np.sum(hist * (np.arange(256) - mean) ** 2)
    skew = np.sum(hist * (np.arange(256) - mean) ** 3) / (var ** 1.5 + 1e-8)
    return {"mean": mean, "var": var, "skew": skew}

# 3.6 基于灰度共生矩阵的纹理特征
def glcm_features(gray_img, distances=[1], angles=[0]):
    # 保证输入为灰度图
    if len(gray_img.shape) == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    ASM = graycoprops(glcm, 'ASM')[0,0]
    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "ASM": ASM
    }

# 3.7 图像的7个Hu不变矩
def hu_moments(gray_img):
    # 保证输入为灰度图
    if len(gray_img.shape) == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return hu

# 4. 主成分分析图像压缩
def pca_compress(img, num_components=20):
    # 保证输入为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    mean, eigenvectors = cv2.PCACompute(gray.reshape(-1, 1).astype(np.float32), mean=None, maxComponents=num_components)
    compressed = np.dot(gray.reshape(-1, 1) - mean, eigenvectors.T)
    reconstructed = np.dot(compressed, eigenvectors) + mean
    out = reconstructed.reshape(gray.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# 形态学操作

def erode(img, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)

def dilate(img, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)

def morph_open(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def morph_close(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def morph_gradient(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def morph_tophat(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def morph_blackhat(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# 轮廓检测扩展

def find_all_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, mode, method)
    return contours, hierarchy

def draw_contour(img, contour, color=(0, 0, 255), thickness=2):
    out = img.copy()
    cv2.drawContours(out, [contour], -1, color, thickness)
    return out

def approx_poly_contour(contour, epsilon_ratio=0.02):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    return approx

def bounding_rect(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h)

def min_area_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box, rect

def min_enclosing_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius

def region_properties(contour):
    # 面积
    area = cv2.contourArea(contour)
    # 周长
    perimeter = cv2.arcLength(contour, True)
    # 质心
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = (cx, cy)
    else:
        centroid = (0, 0)
    # 外接矩形中心
    x, y, w, h = cv2.boundingRect(contour)
    bounding_rect_center = (x + w / 2, y + h / 2)
    # 最小外接圆
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
    min_enclosing_circle_center = (circle_x, circle_y)
    min_enclosing_circle_diameter = 2 * radius
    return {
        "area": area,
        "perimeter": perimeter,
        "centroid": centroid,
        "bounding_rect_center": bounding_rect_center,
        "min_enclosing_circle_center": min_enclosing_circle_center,
        "min_enclosing_circle_diameter": min_enclosing_circle_diameter
    }