import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

if "drawing_mode" not in st.session_state:
    st.session_state['drawing_mode'] = "rect"
if "rect_list" not in st.session_state:
    st.session_state['rect_list'] = [[0] for _ in range(10000)]

if "canvas_info" not in st.session_state:
    st.session_state['canvas_info'] = {}
    
if "img_changed" not in st.session_state:
    st.session_state['img_changed'] = False
    
st.set_page_config(layout = 'wide')
    
    
@st.cache_data
def fetch_and_clean_data(uploaded_file):
    image = Image.open(uploaded_file)
    st.session_state['img_changed'] = True
    print(uploaded_file.file_id)
    return image, uploaded_file.file_id

st.title("이미지에서 사각형 선택 및 좌표 추출")

def confirm_sector_info(image, current_object, odx, canvas_result):
    if f"sector_{odx+1}_already_save" not in st.session_state:
        st.session_state[f'sector_{odx+1}_already_save'] = False
    
    def draw_rect(right, bottom, cropped_image):
        draw = ImageDraw.Draw(cropped_image)
        cropped_width, cropped_height = cropped_image.size
        draw.rectangle((20, 20, right+20, bottom+20), outline="red", width=3)
        draw.rectangle((0, 0, cropped_width, cropped_height), outline="black", width=5)
        return cropped_image, cropped_width, cropped_height
    
    # current_object = canvas_result.json_data['objects'][-1]
    
    left = int(current_object["left"])
    top = int(current_object["top"])
    width = int(current_object["width"])
    height = int(current_object["height"])
    
    cropped_image = image.crop((left-20, top-20, left+width+20, top+height+20))
    result_image, cropped_width, cropped_height = draw_rect(width, height, cropped_image)
    result_image = result_image.resize((int(cropped_width/1.5), int(cropped_height/1.5)))
    with st.expander("화면 보기"):
        st.image(result_image)

    type_dict = {'가로형' : [0, "#00FF00"], '세로형' : [1, "#d60d6e"], '혼합형' : [2, "#0d10d6"]}
    radio_type = st.radio(
                "1. 문항의 형식을 선택하세요.",
                ["가로형", "세로형", "혼합형"],
                index=0,
                horizontal =True,
                key=f'sector_{odx+1}_type'
                )
    type_ = type_dict[radio_type][0]
    
    question_count = st.number_input("2. 섹터 문항 수를 입력하세요.", min_value= 0, value=0, placeholder="필수값이 아닙니다.", key=f'sector_{odx+1}_question_cnt')
    question_counts = [0] * question_count
    for qdx in range(question_count):
        question_counts[qdx] = st.number_input(f"{qdx+1}번째 문항의 보기 수를 작성하세요." ,min_value= 1, key=f"sector_{odx+1}_select{qdx+1}")
        
    if st.session_state[f'sector_{odx+1}_already_save']:
        st.write("**저장되었습니다.**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("저장하기", key=f"sector_{odx+1}_save_btn"):
            if (type_ == 2) & (question_count == 0):
                st.error("혼합형의 문항수, 보기수는 필수입니다.")
            else:
                # st.write("저장되었습니다.")
                st.session_state[f'sector_{odx+1}_already_save'] = True
                question_counts = [type_] + question_counts
                rect_info = [left, top, left+width, top+height, question_counts]
                if rect_info not in st.session_state['rect_list']:
                    st.session_state['rect_list'][odx] = rect_info
                    canvas_result.json_data['objects'][odx]['stroke'] = "#000dff"
                    st.session_state['canvas']['raw']['objects'][odx]['stroke'] = "#000dff"
                    st.session_state['canvas_info'] = canvas_result.json_data
                    st.rerun()
    with col2:    
        if st.button("삭제하기", key=f"sector_{odx+1}_delete_btn"):
            st.write(canvas_result.json_data['objects'][odx])
            # canvas_result.json_data['objects'][odx] = [0]
            canvas_result.json_data['objects'].pop(odx)
            st.session_state['canvas']['raw']['objects'] = canvas_result.json_data['objects']
            st.session_state['canvas_info'] = canvas_result.json_data
            try:
                # st.session_state['rect_list'][odx] = [0]
                st.session_state['rect_list'].pop(odx)
            except Exception as e:
                tmp = 1
            st.session_state[f'sector_{odx+1}_already_save'] = False
            st.rerun()


# 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (JPG 형식)", type=["jpg", "jpeg"])

if uploaded_file is None:
    st.session_state['rect_list'] = [[0] for _ in range(10000)]
    st.session_state['canvas_info'] = {}
    
if uploaded_file is not None:
    col1, col2 = st.columns([0.9,0.1])
    image, file_id= fetch_and_clean_data(uploaded_file)
    if st.session_state['img_changed']:
        st.session_state['rect_list'] = [[0] for _ in range(10000)]
        st.session_state['canvas_info'] = {}
        st.session_state['img_changed'] = False
        st.rerun()
    # try:
    #     st.write(st.session_state['canvas_info'])
    # except Exception as e:
    #     st.write(e)
    drawing_mode = 'rect'
    stroke_width = 2
    # stroke_color = type_dict[radio_type][1]
    # bg_image = Image.open(uploaded_file)
    original_width, original_height = image.size
    # 스트림릿 캔버스 생성
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            # stroke_color=stroke_color,
            stroke_color="#00ff00",
            background_image=image,
            # background_image=resized_image,
            update_streamlit=True,
            height=image.height,
            width=image.width,
            # height=new_height,
            # width=new_width,
            display_toolbar = True,
            drawing_mode='rect',
            initial_drawing = st.session_state['canvas_info'] if st.session_state['canvas_info'] != {} else {},
            key="canvas",
        )
        # st.write(canvas_result.json_data)
        
        # canvas_result = st_canvas()
        # print(canvas_result.json_data)
        # canvas_result.json_data['objects'] = canvas_result.json_data['objects'][:-1]
        # st_canvas(initial_drawing=canvas_result.json_data)

    with col2:
        if (canvas_result.json_data is not None):
            if (canvas_result.json_data['objects'] != []):
                for odx, obj in enumerate(st.session_state['canvas']['raw']['objects']):
                    with st.popover(f"{odx+1}번째 섹터", use_container_width =True):
                        confirm_sector_info(image, obj, odx, canvas_result)
        st.divider()
        if st.button("페이지 확정"):
            final_rect = [i for i in st.session_state['rect_list'] if i != [0]]
            with st.container(border=1):
                st.code(final_rect)
        st.write([i for i in st.session_state['rect_list'] if i != [0]])
        st.write(st.session_state['canvas'])