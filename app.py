import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Streamlit 웹페이지 제목
st.title("이미지에서 사각형 선택 및 좌표 추출")

# 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (JPG 형식)", type=["jpg", "jpeg"])

# 이미지가 업로드된 경우
if uploaded_file is not None:
    # 이미지를 열고 스트림릿에 표시
    image = Image.open(uploaded_file)
    # st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    # 캔버스 설정
    drawing_mode = "rect"
    stroke_width = 2
    stroke_color = "#00FF00"
    bg_image = Image.open(uploaded_file)
    
    # 스트림릿 캔버스 생성
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=bg_image,
        update_streamlit=True,
        height=bg_image.height,
        width=bg_image.width,
        display_toolbar = True,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # 사각형 좌표 출력
    rects_point = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        for obj in objects:
            if obj["type"] == "rect":
                left = obj["left"]
                top = obj["top"]
                width = obj["width"]
                height = obj["height"]
                st.write(f"사각형의 꼭지점 좌표:")
                st.write(f"왼쪽 위: ({left}, {top})")
                st.write(f"오른쪽 아래: ({left + width}, {top + height})")
                rects_point.append([left, top, left + width, top + height])
                st.code(rects_point)
