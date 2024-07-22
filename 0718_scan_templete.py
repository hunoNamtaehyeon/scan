import cv2
import pandas as pd
import numpy as np
import os
import math
import json

from pdf2image import convert_from_path
from PIL import Image
from matplotlib import pyplot as plt

origin_pdf_dir = './scan/datas/240718_좌표용_testPDF/300dpi'   # PDF 파일이 있는 폴더 경로
pdf_to_divied_jpg_dir = origin_pdf_dir + "/divided"              # pdf를 n개 쪽의 jpg로 저장할(된) 디렉토리
divied_jpg_crop_from_vertex_dir = origin_pdf_dir + "/square"    # n개쪽의 jpg들을 꼭지점 네모를 기준으로 잘라 저장할(된) 디렉토리
draw_sertors_and_circles_dir = origin_pdf_dir + "/circles"      # 동그라미 검출 이미지를 저장할 폴더 경로

project_dict = {'5537' : "MIT(중등)",
                '5539' : "MIT(고등)",
                '5541' : "CLS(초등)",
                '5543' : "CLS(중등)",
                '5545' : "POWER",
                '5547' : "FIT"}

########################################################################################################

# def cover_edges_with_white(image, border_size):
#     # OpenCV는 이미지를 numpy 배열로 처리
#     image_np = np.array(image)
    
#     # 현재 이미지의 크기
#     height, width = image_np.shape[:2]
    
#     # 상단 가장자리 덮기
#     image_np[:border_size, :] = [255, 255, 255]
#     # 하단 가장자리 덮기
#     image_np[height-border_size:, :] = [255, 255, 255]
#     # 왼쪽 가장자리 덮기
#     image_np[:, :border_size] = [255, 255, 255]
#     # 오른쪽 가장자리 덮기
#     image_np[:, width-border_size:] = [255, 255, 255]
    
#     return image_np

# def is_square(cnt, min_area, max_area):
#     area = cv2.contourArea(cnt)
#     if area < min_area or area > max_area:
#         return False

#     return True

# def add_white_border(image, border_size):
#     # OpenCV는 이미지를 numpy 배열로 처리
#     image_np = np.array(image)
    
#     # 하얀색 여백 추가 (위, 아래, 왼쪽, 오른쪽)
#     bordered_image = cv2.copyMakeBorder(image_np, 
#                                         border_size, border_size, border_size, border_size, 
#                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     return bordered_image










# def convert_pdf_folder_to_images(pdf_folder, output_folder, project_dict):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
        
#     project_dict = {'5537' : "MIT(중등)",
#                     '5539' : "MIT(고등)",
#                     '5541' : "CLS(초등)",
#                     '5543' : "CLS(중등)",
#                     '5545' : "POWER",
#                     '5547' : "FIT"}
    
#     for filename in os.listdir(pdf_folder):
#         if filename.endswith('.pdf'):
#             project_name = project_dict[filename.replace(".pdf", "")]
#             print(project_name)
#             pdf_path = os.path.join(pdf_folder, filename)
#             images = convert_from_path(pdf_path)
#             base_filename = os.path.splitext(filename)[0]
            
#             border_size = 30
#             covered_images = [cover_edges_with_white(image, border_size) for image in images]
            
#             for page_num, img in enumerate(covered_images):
#                 if page_num == 0:
#                     left_num = len(images)*2
#                     right_num = 1
#                 else:
#                     left_num = right_num + 1
#                     right_num = left_num + 1
                
#                 height, width = img.shape[:2]
#                 mid_width = width // 2
                
#                 left_img = img[:, :mid_width]
#                 right_img = img[:, mid_width:]
                
#                 output_path = os.path.join(output_folder, f"{base_filename}_origin_{page_num + 1}.jpg")
#                 left_path = os.path.join(output_folder, f"{base_filename}_{left_num}.jpg")
#                 right_path = os.path.join(output_folder, f"{base_filename}_{right_num}.jpg")
                
#                 # cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#                 # cv2.imwrite(left_path, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
#                 # cv2.imwrite(right_path, cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
#                 for path, image in zip([output_path, left_path, right_path], [img, left_img, right_img]):
#                     cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#                     image_pil = Image.open(path)
#                     image_pil.save(path, dpi=(300, 300))
#             print()
            


# convert_pdf_folder_to_images(origin_pdf_dir, pdf_to_divied_jpg_dir, project_dict) # chk : True -> 마킹된 검사지 

########################################################################################################
########################################################################################################


    
def draw_contours(image_folder, output_folder, min_area=100, max_area=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    bordered_folder = output_folder+"_bordered"
    if not os.path.exists(bordered_folder):
        os.makedirs(bordered_folder)
    
    cut_img_dict = dict()
    for filename in os.listdir(image_folder):
        if (filename.endswith('.jpg')) and ("_origin" not in filename):
            for_path = filename.replace(".jpg", "")
            splited_for_path = for_path.split("_")
            page_num = splited_for_path[-1]
            pj_name = "_".join(splited_for_path[:-1])
            if pj_name not in cut_img_dict:
                cut_img_dict[pj_name] = []
                
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 경계 검출 (Canny Edge Detection)
            edges = cv2.Canny(blurred, 50, 100) # 50 100

            # 윤곽선 검출
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cnt for cnt in contours if is_square(cnt, min_area, max_area)]
            rectangles = rectangles[:10] + rectangles[-10:]

            top_left = (float('inf'), float('inf'))
            top_right = (float('-inf'), float('inf'))
            bottom_left = (float('inf'), float('-inf'))
            bottom_right = (float('-inf'), float('-inf'))

            cut_xy = [0, 0, 0, 0]
            for contour in rectangles:
                x, y, w, h = cv2.boundingRect(contour)
                points = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
                
                for point in points:
                    if point[0] + point[1] < top_left[0] + top_left[1]:
                        top_left = point
                        cut_xy[:2] = point
                    if point[0] - point[1] > top_right[0] - top_right[1]:
                        top_right = point
                    if point[0] - point[1] < bottom_left[0] - bottom_left[1]:
                        bottom_left = point
                    if point[0] + point[1] > bottom_right[0] + bottom_right[1]:
                        bottom_right = point
                        cut_xy[2:] = point
                        
            original_points = np.float32([top_left, top_right, bottom_left, bottom_right])
            dst_points = np.float32([[0, 0], [800, 0], [0, 1200], [800, 1200]])
            matrix = cv2.getPerspectiveTransform(original_points, dst_points)
            transformed_image = cv2.warpPerspective(img, matrix, (800, 1200))
            
            result_image_path = os.path.join(output_folder, f"{filename}")
            # cv2.imwrite(result_image_path, transformed_image)
            image_pil = Image.open(result_image_path)
            image_pil.save(result_image_path, dpi=(300, 300))
            
            bordered_result_image_path = os.path.join(bordered_folder, f"{filename}")
            bordered_image = add_white_border(transformed_image, 30)
            # cv2.imwrite(bordered_result_image_path, bordered_image)
            image_pil = Image.open(bordered_result_image_path)
            image_pil.save(bordered_result_image_path, dpi=(300, 300))

    return cut_img_dict
# cut_img_dict = draw_contours(pdf_to_divied_jpg_dir, divied_jpg_crop_from_vertex_dir)

########################################################################################################
########################################################################################################


# origin_pdf_dir = './scan/datas/240718_좌표용_testPDF/300dpi'   # PDF 파일이 있는 폴더 경로
# pdf_to_divied_jpg_dir = origin_pdf_dir + "/divided"              # pdf를 n개 쪽의 jpg로 저장할(된) 디렉토리
# divied_jpg_crop_from_vertex_dir = origin_pdf_dir + "/square"    # n개쪽의 jpg들을 꼭지점 네모를 기준으로 잘라 저장할(된) 디렉토리
# draw_sertors_and_circles_dir = origin_pdf_dir + "/circles"      # 동그라미 검출 이미지를 저장할 폴더 경로

def cover_edges_with_white(image, border_size):
    # OpenCV는 이미지를 numpy 배열로 처리
    image_np = np.array(image)
    
    # 현재 이미지의 크기
    height, width = image_np.shape[:2]
    
    # 상단 가장자리 덮기
    image_np[:border_size, :] = [255, 255, 255]
    # 하단 가장자리 덮기
    image_np[height-border_size:, :] = [255, 255, 255]
    # 왼쪽 가장자리 덮기
    image_np[:, :border_size] = [255, 255, 255]
    # 오른쪽 가장자리 덮기
    image_np[:, width-border_size:] = [255, 255, 255]
    
    return image_np

def is_square(cnt, min_area, max_area):
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        return False

    return True

def add_white_border(image, border_size):
    # OpenCV는 이미지를 numpy 배열로 처리
    image_np = np.array(image)
    
    # 하얀색 여백 추가 (위, 아래, 왼쪽, 오른쪽)
    bordered_image = cv2.copyMakeBorder(image_np, 
                                        border_size, border_size, border_size, border_size, 
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered_image

def convert_pdf_folder_to_images(origin_pdf_dir, pdf_to_divied_jpg_dir, divied_jpg_crop_from_vertex_dir):
    if not os.path.exists(pdf_to_divied_jpg_dir):
        os.makedirs(pdf_to_divied_jpg_dir)
    if not os.path.exists(divied_jpg_crop_from_vertex_dir):
        os.makedirs(divied_jpg_crop_from_vertex_dir)
    bordered_folder = divied_jpg_crop_from_vertex_dir+"_bordered"
    if not os.path.exists(bordered_folder):
        os.makedirs(bordered_folder)
        
    project_dict = {'5537' : "MIT(중등)",
                    '5539' : "MIT(고등)",
                    '5541' : "CLS(초등)",
                    '5543' : "CLS(중등)",
                    '5545' : "POWER",
                    '5547' : "FIT"}
    for filename in os.listdir(origin_pdf_dir):
        if filename.endswith('.pdf'): # 5547.pdf
            project_name = project_dict[filename.replace(".pdf", "")] # FIT, POWER ...
            pdf_path = os.path.join(origin_pdf_dir, filename) # ./scan/datas/240718_좌표용_testPDF/300dpi/5547.pdf
            images = convert_from_path(pdf_path)
            base_filename = os.path.splitext(filename)[0] # 5547
            
            border_size = 30
            covered_images = [cover_edges_with_white(image, border_size) for image in images]
            print(pdf_path)

            for page_num, img in enumerate(covered_images):
                if page_num == 0:
                    left_num = len(images)*2
                    right_num = 1
                else:
                    left_num = right_num + 1
                    right_num = left_num + 1
                
                height, width = img.shape[:2]
                mid_width = width // 2
                
                left_img = img[:, :mid_width]
                right_img = img[:, mid_width:]
                
                output_path = os.path.join(pdf_to_divied_jpg_dir, f"{base_filename}_origin_{page_num + 1}.jpg")
                left_path = os.path.join(pdf_to_divied_jpg_dir, f"{base_filename}_{left_num}.jpg")
                right_path = os.path.join(pdf_to_divied_jpg_dir, f"{base_filename}_{right_num}.jpg")
                
                for path, image in zip([output_path, left_path, right_path], [img, left_img, right_img]):
                    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    image_pil = Image.open(path)
                    image_pil.save(path, dpi=(300, 300))
                    
                min_area=100
                max_area=1000
                for img, img_num in zip([left_img, right_img], [left_num, right_num]):
                    height, width = img.shape[:2]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                    # 경계 검출 (Canny Edge Detection)
                    edges = cv2.Canny(blurred, 50, 100) # 50 100

                    # 윤곽선 검출
                    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    rectangles = [cnt for cnt in contours if is_square(cnt, min_area, max_area)]
                    rectangles = rectangles[:10] + rectangles[-10:]

                    top_left = (float('inf'), float('inf'))
                    top_right = (float('-inf'), float('inf'))
                    bottom_left = (float('inf'), float('-inf'))
                    bottom_right = (float('-inf'), float('-inf'))

                    cut_xy = [0, 0, 0, 0]
                    for contour in rectangles:
                        x, y, w, h = cv2.boundingRect(contour)
                        points = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
                        
                        for point in points:
                            if point[0] + point[1] < top_left[0] + top_left[1]:
                                top_left = point
                                cut_xy[:2] = point
                            if point[0] - point[1] > top_right[0] - top_right[1]:
                                top_right = point
                            if point[0] - point[1] < bottom_left[0] - bottom_left[1]:
                                bottom_left = point
                            if point[0] + point[1] > bottom_right[0] + bottom_right[1]:
                                bottom_right = point
                                cut_xy[2:] = point
                    kernel = np.array([[0, -1, 0],
                                        [-1, 5, -1],
                                        [0, -1, 0]])            
                    original_points = np.float32([top_left, top_right, bottom_left, bottom_right])
                    
                    # height, width = 1200, 800
                    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                    matrix = cv2.getPerspectiveTransform(original_points, dst_points)
                    transformed_image = cv2.warpPerspective(img, matrix, (width, height))
                    # transformed_image = cv2.filter2D(transformed_image, -1, kernel)
                    
                    for idx in range(2):
                        project_num = filename.split(".")[0]
                        if idx == 0: # for_canvas
                            project_num += "_for_canvas"
                            image = cv2.resize(transformed_image, dsize=(0,0), fx=0.5, fy=0.5,
                                                        interpolation=cv2.INTER_LANCZOS4)
                        else: # origin
                            image = transformed_image.copy()
                            image = cv2.filter2D(image, -1, kernel)
                                
                        fn = f"{project_num}_{img_num}.jpg"
                        result_image_path = os.path.join(divied_jpg_crop_from_vertex_dir, f"{fn}")
                        bordered_result_image_path = os.path.join(bordered_folder, f"{fn}")
                        
                        blurred = cv2.GaussianBlur(image, (7, 7), 2.0)

                        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
                        
                        cv2.imwrite(result_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        
                        bordered_image = add_white_border(image, 30)
                        cv2.imwrite(bordered_result_image_path, cv2.cvtColor(bordered_image, cv2.COLOR_RGB2BGR), 
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    
                    
                    
                    
                    
                    
                    # transformed_image = cv2.resize(transformed_image, dsize=(0,0), fx=0.5, fy=0.5,
                    #                                interpolation=cv2.INTER_LANCZOS4)
                    
                    # # 가우시안 블러 적용
                    # blurred = cv2.GaussianBlur(transformed_image, (7, 7), 2.0)

                    # # 원본 이미지에서 블러 처리된 이미지를 빼서 엣지 강조
                    # transformed_image = cv2.addWeighted(transformed_image, 1.5, blurred, -0.5, 0)
                    # # transformed_image = cv2.filter2D(transformed_image, -1, kernel)
                    
                    # fn = filename.split(".")[0]
                    # fn = f"{fn}_{img_num}.jpg"
                    # result_image_path = os.path.join(divied_jpg_crop_from_vertex_dir, f"{fn}")
                    # cv2.imwrite(result_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR), 
                    #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    
                    # bordered_result_image_path = os.path.join(bordered_folder, f"{fn}")
                    # bordered_image = add_white_border(transformed_image, 30)
                    # cv2.imwrite(bordered_result_image_path, cv2.cvtColor(bordered_image, cv2.COLOR_RGB2BGR), 
                    #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # print()
            
            
convert_pdf_folder_to_images(origin_pdf_dir, pdf_to_divied_jpg_dir, divied_jpg_crop_from_vertex_dir)