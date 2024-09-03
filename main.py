import pygame
from math import cos, pi
import numpy as np
import os, glob
import cv2



class Heart:
	def __init__(self, title="I Love You", frame_num=20, seed_points_num=2000, seed_num: int=None, highlight_rate=0.3,
				 background_img_dir="", set_bg_imgs=False, bg_weight=0.3, curve_weight=0.7, frame_width=1080, frame_height=960, scale=10.1,
				 base_color: tuple[int, int, int]=None, highlight_points_color_1: tuple[int, int, int]=None, highlight_points_color_2: tuple[int, int, int]=None, wait=100):
		
		pygame.init()
		pygame.font.init()
		self.font = pygame.font.Font("AlexBrush-Regular.ttf", 100)
		pygame.display.set_caption(title)
		
		np.random.seed(seed_num)

		self.title: str = title
		self.highlight_points_color_2: tuple[int, int, int] = highlight_points_color_2
		self.highlight_points_color_1: tuple[int, int, int] = highlight_points_color_1
		self.highlight_rate: float = highlight_rate
		self.base_color: tuple[int, int, int] = base_color

		self.curve_weight: float = curve_weight
		img_paths: list[str] = glob.glob(background_img_dir + "/*")
		self.bg_imgs: list[pygame.surface.Surface] = []
		self.set_bg_imgs: bool = set_bg_imgs
		self.bg_weight: float = bg_weight
		if os.path.exists(background_img_dir) and len(img_paths) > 0 and set_bg_imgs:
			for img_path in img_paths:
				img = cv2.imread(img_path)
				self.bg_imgs.append(img)
			first_bg = self.bg_imgs[0]
			# width = int(first_bg.shape[1] * bg_img_scale)
			# height = int(first_bg.shape[0] * bg_img_scale)

			first_bg = cv2.resize(first_bg, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

			new_bg_imgs = []
			for img in self.bg_imgs:

				img = cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

				new_bg_imgs.append(img)
			
			self.bg_imgs = new_bg_imgs
			# assert all(img.shape[0] == first_bg.shape[0] and img.shape[1] == first_bg.shape[1] for img in self.bg_imgs)
			self.frame_width = self.bg_imgs[0].shape[1]
			self.frame_height = self.bg_imgs[0].shape[0]
		else:
			self.frame_width = frame_width 
			self.frame_height = frame_height 
		
		self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))

		self.center_x: int = self.frame_width // 2
		self.center_y: int = self.frame_height // 2
		self.main_curve_width = -1
		self.main_curve_height = -1

		self.frame_points: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
		self.frame_num = frame_num
		self.seed_num: int = seed_num 
		self.seed_points_num: int = seed_points_num
		self.scale: float = scale 
		self.wait: int = wait
	
	def heart_function(self, t: np.ndarray, frame_idx=0, scale=5.20) -> tuple[np.ndarray, np.ndarray]:
		trans = 3 - (1 + self.periodic_func(frame_idx, self.frame_num)) * 0.5 

		x = 15 * (np.sin(t) ** 3)
		t = np.where((pi < t) & (t < 2 * pi), 2 * pi - t, t)
		y = -(14 * np.cos(t) - 4 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(trans * t))

		ign_area = 0.15
		center_ids = np.where((x > -ign_area) & (x < ign_area))

		if np.random.random() > 0.32:
			x, y = np.delete(x, center_ids), np.delete(y, center_ids) 

		x *= scale
		y *= scale

		x += self.center_x
		y += self.center_y

		return x.astype(int), y.astype(int)

	def shrink(self, x: np.ndarray, y: np.ndarray, ratio: float, offset=1, p=0.5, dist_func="uniform") -> tuple[np.ndarray, np.ndarray]:
		x_ = (x - self.center_x)
		y_ = (y - self.center_y)
		force = 1 / ((x_ ** 2 + y_ ** 2) ** p + 1e-30)

		dx = ratio * force * x_
		dy = ratio * force * y_

		def d_offset(x):
			if dist_func == "uniform":
				return x + np.random.uniform(-offset, offset, size=x.shape)
			elif dist_func == "norm":
				return x + offset * np.random.normal(0, 1, size=x.shape)

		dx, dy = d_offset(dx), d_offset(dy)

		return x - dx, y - dy

	def scatter(self, x: np.ndarray, y: np.ndarray, alpha=0.75, beta=0.15) -> tuple[np.ndarray, np.ndarray]:
		ratio_x = - beta * np.log(np.random.random(x.shape) * alpha)
		ratio_y = - beta * np.log(np.random.random(y.shape) * alpha)
		dx = ratio_x * (x - self.center_x)
		dy = ratio_y * (y - self.center_y)

		return x - dx, y - dy

	def periodic_func(self, x: float, x_num: int) -> float:
		def ori_func(t):
			return cos(t)

		func_period = 2 * pi
		return ori_func(x / x_num * func_period)

	def gen_points(self, points_num: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		cy = self.periodic_func(frame_idx, self.frame_num)
		ratio = 10 * cy

		period =  2 * pi
		seed_points = np.linspace(0, period, points_num)
		seed_x, seed_y = self.heart_function(seed_points, frame_idx, scale=self.scale)
		x, y = self.shrink(seed_x, seed_y, ratio, offset=2)
		curve_width, curve_height = int(x.max() - x.min()), int(y.max() - y.min())
		self.main_curve_width = max(self.main_curve_width, curve_width)
		self.main_curve_height = max(self.main_curve_height, curve_height)
		point_size = np.random.choice([1, 2], x.shape, replace=True, p=[0.5, 0.5])
		tag = np.ones_like(x)


		def delete_points(x_, y_, ign_area, ign_prop):
			center_ids = np.where((x_ > self.center_x - ign_area) & (x_ < self.center_x + ign_area))
			center_ids = center_ids[0]
			np.random.shuffle(center_ids)
			del_num = round(len(center_ids) * ign_prop)
			del_ids = center_ids[:del_num]
			x_, y_ = np.delete(x_, del_ids), np.delete(y_, del_ids)
			return x_, y_

		for idx, beta in enumerate(np.linspace(0.05, 0.2, 6)):
			alpha = 1 - beta
			x_, y_ = self.scatter(seed_x, seed_y, alpha, beta)
			x_, y_ = self.shrink(x_, y_, ratio, offset=round(beta * 15))
			x = np.concatenate((x, x_), 0)
			y = np.concatenate((y, y_), 0)
			p_size = np.random.choice([1, 2], x_.shape, replace=True, p=[0.55 + beta, 0.45 - beta])
			point_size = np.concatenate((point_size, p_size), 0)
			tag_ = np.ones_like(x_) * 2
			tag = np.concatenate((tag, tag_), 0)
		

		halo_ratio = int(7 + 2 * abs(cy)) 

		x_, y_ = self.heart_function(seed_points, frame_idx, scale=self.scale + 0.9)
		x_1, y_1 = self.shrink(x_, y_, halo_ratio, offset=18, dist_func="uniform")
		x_1, y_1 = delete_points(x_1, y_1, 20, 0.5)
		x = np.concatenate((x, x_1), 0)
		y = np.concatenate((y, y_1), 0)

		halo_number = int(points_num * 0.6 + points_num * abs(cy))
		seed_points = np.random.uniform(0, 2 * pi, halo_number)
		x_, y_ = self.heart_function(seed_points, frame_idx, scale=self.scale + 0.9)
		x_2, y_2 = self.shrink(x_, y_, halo_ratio, offset=int(6 + 15 * abs(cy)), dist_func="norm")
		x_2, y_2 = delete_points(x_2, y_2, 20, 0.5)
		x = np.concatenate((x, x_2), 0)
		y = np.concatenate((y, y_2), 0)
		x_3, y_3 = self.heart_function(np.linspace(0, 2 * pi, int(points_num * .4)),
											 frame_idx, scale=self.scale + 0.2)
		x_3, y_3 = self.shrink(x_3, y_3, ratio * 2, offset=6)
		x = np.concatenate((x, x_3), 0)
		y = np.concatenate((y, y_3), 0)


		halo_len = x_1.shape[0] + x_2.shape[0] + x_3.shape[0]
		p_size = np.random.choice([1, 2, 3], halo_len, replace=True, p=[0.7, 0.2, 0.1])
		point_size = np.concatenate((point_size, p_size), 0)
		tag_ = np.ones(halo_len) * 2 * 3

		tag = np.concatenate((tag, tag_), 0)
		

		x_y = np.around(np.stack([x, y], axis=1), 0)
		x, y = x_y[:, 0], x_y[:, 1]
		return x, y, point_size, tag

	def get_frames(self) -> list[np.ndarray]:
		for frame_idx in range(self.frame_num):
			np.random.seed(self.seed_num)
			self.frame_points.append(self.gen_points(self.seed_points_num, frame_idx))
		

		frames: list[np.ndarray] = []

		def add_points(frame, x, y, size, tag):
			highlight1 = np.array(self.highlight_points_color_1, dtype='uint8')
			highlight2 = np.array(self.highlight_points_color_2, dtype='uint8')
			base_col = np.array(self.base_color, dtype='uint8')

			x, y = x.astype(int), y.astype(int)
			frame[y, x] = base_col

			size_2 = np.int64(size == 2)
			frame[y, x + size_2] = base_col
			frame[y + size_2, x] = base_col

			size_3 = np.int64(size == 3)
			frame[y + size_3, x] = base_col
			frame[y - size_3, x] = base_col
			frame[y, x + size_3] = base_col
			frame[y, x - size_3] = base_col
			frame[y + size_3, x + size_3] = base_col
			frame[y - size_3, x - size_3] = base_col
			# frame[y - size_3, x + size_3] = base_col
			# frame[y + size_3, x - size_3] = base_col

			random_sample = np.random.choice([1, 0], size=tag.shape, p=[self.highlight_rate, 1 - self.highlight_rate])

			# tag2_size1 = np.int64((tag <= 2) & (size == 1) & (random_sample == 1))
			# frame[y * tag2_size1, x * tag2_size1] = highlight2

			tag2_size2 = np.int64((tag <= 2) & (size == 2) & (random_sample == 1))
			# print(tag.shape, size.shape)
			# print(tag2_size2)
			frame[y * tag2_size2, x * tag2_size2] = highlight1
			# frame[y * tag2_size2, (x + 1) * tag2_size2] = highlight2
			# frame[(y + 1) * tag2_size2, x * tag2_size2] = highlight2
			frame[(y + 1) * tag2_size2, (x + 1) * tag2_size2] = highlight2

		for x, y, size, tag in self.frame_points:
			frame = np.zeros([self.frame_height, self.frame_width, 3], dtype="uint8")
			add_points(frame, x, y, size, tag)
			frames.append(frame)

		return frames
	
	def draw(self, frame):
		self.screen.fill((0, 0, 0))

		self.screen.blit(pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "BGR"), (0, -100))

		text = self.font.render("Nguyễn Trúc Quỳnh", True, (self.base_color))

		text_rect = text.get_rect(center=(self.center_x, self.frame_height - 150))

		self.screen.blit(text, text_rect)

		pygame.display.update()
	
	def run(self):
		frames = self.get_frames()

		for frame in frames:
			frame[:,:,[1,2,0]] = frame[:,:,[2,0,1]]


		run = True

		count = 0
		while run:
			if len(self.bg_imgs) > 0:
				count %= len(self.bg_imgs)

			for frame in frames:
				pygame.time.delay(self.wait)
				if len(self.bg_imgs) > 0 and self.set_bg_imgs:
					frame = cv2.addWeighted(self.bg_imgs[int(count)],self.bg_weight,frame,self.curve_weight,0)

				self.draw(frame)

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						run = False
			
			count += 0.5

		
		pygame.quit()

def main():
	import yaml
	settings = yaml.load(open("./settings.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
	if settings["wait"] == -1:
		settings["wait"] = int(settings["period_time"] / settings["frame_num"])
	del settings["period_time"]
	del settings["times"]
	heart = Heart(seed_num=15082004, **settings)
	heart.run()

if __name__ == "__main__":
	main()
