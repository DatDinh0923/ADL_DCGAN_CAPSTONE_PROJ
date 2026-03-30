from PIL import Image

def combine_images_horizontally(image_paths, output_path, padding=10):
    """
    Ghép các ảnh theo chiều ngang thành một bức ảnh duy nhất.
    
    :param image_paths: Danh sách đường dẫn các ảnh cần ghép
    :param output_path: Đường dẫn file ảnh đầu ra
    :param padding: Khoảng cách (số pixel màu trắng) giữa các ảnh
    """
    # Mở tất cả các ảnh
    images = [Image.open(x) for x in image_paths]
    
    # Tính toán kích thước cho bức ảnh tổng (chiều dài = tổng chiều dài các ảnh + padding)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)

    # Tạo một bức ảnh mới trống (nền trắng)
    new_im = Image.new('RGB', (total_width, max_height), color='white')

    # Dán từng ảnh vào bức ảnh tổng
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width + padding # Dịch sang phải để dán ảnh tiếp theo

    # Lưu ảnh kết quả
    new_im.save(output_path)
    print(f"✅ Đã ghép thành công và lưu tại: {output_path}")

# --- CẤU HÌNH ---
# Điền đúng đường dẫn tới các ảnh epoch của bạn
images_to_combine = [
    '/home/insomnia/1Code_Workspace/ADL/output_w_SN_DO_128px_bs64/fake_samples_epoch_0_iter_0.png',
    '/home/insomnia/1Code_Workspace/ADL/output_w_SN_DO_128px_bs64/fake_samples_epoch_5_iter_18500.png',
    '/home/insomnia/1Code_Workspace/ADL/output_w_SN_DO_128px_bs64/fake_samples_epoch_7_iter_23500.png',
    '/home/insomnia/1Code_Workspace/ADL/output_w_SN_DO_128px_bs64/fake_samples_epoch_9_iter_31659.png'
]

# Tên file ảnh đầu ra
output_filename = 'images/epoch_evolution_combined_128px_bs64_SN_DO.png'

# Gọi hàm chạy (padding=10 để tạo khoảng trắng mỏng giữa các ảnh cho đẹp)
combine_images_horizontally(images_to_combine, output_filename, padding=10)