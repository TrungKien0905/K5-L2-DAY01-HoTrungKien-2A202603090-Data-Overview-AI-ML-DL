# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Vì: 
    + Mô hình YOLO11 Classification nhận đầu vào là toàn bộ hình ảnh.
    + Mô hình đưa ra danh sách các lớp có khả năng mô tả nội dung của bức ảnh.
    + Kết quả không chứa tọa độ Bounding Box hoặc vị trí cụ thể của từng vật thể.
    + Trong ảnh giao thông có nhiều phương tiện như taxi, xe buýt, ô tô và xe tải, nhưng mô hình chỉ đưa ra các lớp dự đoán đại diện cho toàn bộ ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp không phải do mô hình tự nghĩ ra mà được xác định bởi taxonomy đi kèm với bộ dữ liệu và checkpoint đã được huấn luyện.
- Khi mô hình thực hiện dự đoán, nó tính toán điểm số cho các lớp đã được xác định trước trong quá trình huấn luyện, sau đó sắp xếp kết quả theo thứ tự điểm số.
- Kết luận: Model chỉ dự đoán trong phạm vi các lớp mà nó đã được huấn luyện, không tự tạo ra những lớp mới khi thực hiện inference.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần lưu trữ cả ba trường để đảm bảo tính chính xác, nhất quán và khả năng truy xuất dữ liệu trong quá trình quản lý dataset.
Trường	        Mục đích
class_id	    Định danh lớp bằng mã số, thuận tiện cho xử lý dữ liệu và huấn luyện mô hình.
class_name	    Giúp con người đọc và hiểu ý nghĩa của lớp.
taxonomy_name	Xác định hệ thống phân loại mà lớp đó thuộc về.     
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ nguyên tắc lựa chọn lớp đại diện cho toàn bộ hình ảnh, đặc biệt trong trường hợp ảnh chứa nhiều vật thể hoặc nhiều chủ đề khác nhau.
Một số nội dung cần được quy định:
    1. Đối tượng chính: Xác định vật thể hoặc chủ đề quan trọng nhất trong bức ảnh.
    2. Mức độ chiếm ưu thế: Xem xét đối tượng nào chiếm diện tích lớn hoặc thu hút sự chú ý nhiều nhất.
    3. Ngữ cảnh hình ảnh: Xem xét mục đích và nội dung tổng thể của bức ảnh.
    4. Quy tắc ưu tiên: Nếu có nhiều đối tượng cùng xuất hiện, cần có quy tắc ưu tiên để chọn một lớp duy nhất.
    5. Trường hợp không rõ ràng: Quy định cách xử lý khi không thể xác định lớp đại diện, chẳng hạn chuyển sang trạng thái cần review.
6. Vì sao không được lấy `score` của model làm ground truth?
- Không được sử dụng trực tiếp score của mô hình làm Ground Truth vì đây chỉ là điểm số thể hiện mức độ mô hình ưu tiên một lớp dự đoán, không phải nhãn chính xác đã được con người xác minh.
- Có một số lý do:
    1. Model có thể dự đoán sai: Lớp có score cao nhất chưa chắc đã phản ánh đúng nội dung hình ảnh.
    2. Score không phải độ chính xác: Ví dụ, score = 0.510915 không có nghĩa mô hình chắc chắn đúng 51,09% theo nghĩa xác suất chính xác đã được kiểm chứng.
    3. Có thể tồn tại nhiều đối tượng: Một hình ảnh có thể chứa nhiều vật thể, nhưng mô hình Classification chỉ đưa ra các lớp dự đoán cho toàn bộ ảnh.
    4. Cần có sự xác minh: Ground Truth phải được xác định dựa trên dữ liệu thực tế và guideline gán nhãn, sau đó được kiểm tra chất lượng.
- Kết luận: Score chỉ là thông tin tham khảo từ mô hình. Ground Truth phải được xác định thông qua quá trình gán nhãn và kiểm tra bởi con người theo guideline đã thống nhất.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= oven ; `score`= 0.686791 ; `bbox_xyxy`=[0.11, 187.54, 195.96, 292.81] ; `bbox_width`= 195.85 ; `bbox_height`= 105.26
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Ảnh rộng 640 và cao 427 pixel, gốc (0,0) ở góc trên-trái ảnh. 
- Góc trên-trái: cách mép trái 0.11 pixel và cách mép trên 187 pixel. 
- Góc dưới-phải: cách mép trái 196 pixel và cách mép trên 293 pixel. 
- Kích thước: rộng khoảng 196 pixel và cao khoảng 105 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng	    Số prediction	    Ghi chú
0.20	    17	                Thêm 6 hộp so với 0.35
0.35	    11	
0.60	    6	                Mất 5 hộp so với 0.35
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Hạ ngưỡng giữ lại thêm nhiều hộp điểm thấp, nên reviewer phải:
    + Kiểm tra từng hộp thêm này xem có đúng vật không, rồi xóa hộp sai (ví dụ cái nồi bị gọi là bowl).
    + Kiểm tra lại nhãn lớp, vì điểm thấp thường kèm nhầm lớp (cup và bowl).
    + Loại hộp trùng nhau, cùng bao một vật.
    + Chỉnh lại những hộp lệch hoặc quá rộng.
- Đổi lại, ít sót vật hơn, nhưng khối lượng review và thời gian tăng lên.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Mỗi vật là một box. Box phải ôm sát phần nhìn thấy của vật: cạnh box chạm điểm ngoài cùng của vật ở bốn phía, không chừa nền thừa và không cắt mất phần vật. Không vẽ hai box cho cùng một vật.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Ví dụ: person 0.61, box [0.12, 263.18, 61.55, 310.78], nằm sát mép trái ảnh và chỉ thấy phần tay. Hộp bowl 0.38 ở mép trái cũng bị cắt.
- Tôi sẽ viết thành quy tắc trong guideline. Lý do:
    + Vật bị cắt ở mép ảnh gặp ở rất nhiều ảnh, nên cần một cách làm thống nhất để mọi người gán nhãn cho cùng kết quả.
    + Quy tắc cắt mép trả lời được dứt khoát: chỉ box phần nhìn thấy, không đoán phần ngoài khung ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-002 ; `class_name`= bowl ; `score`= 0.735744 ; số điểm polygon=`polygon_point_count`= 67 ; 5 điểm đầu của `polygon_xy`= [53, 344], [52, 345], [50, 345], [49, 346], [48, 346]
- Box của record này là [31.44, 343.77, 100.89, 385.55].
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Box chỉ cho biết vật nằm trong một hình chữ nhật, rộng khoảng 69 và cao khoảng 42 pixel. Polygon cho biết viền thật của vật. Với cái bát này, polygon đi từ đỉnh (53, 344) xuống góc trái (32, 354), xuống đáy (44 đến 66, 385), rồi lên mép phải (99, 348). Các góc của box, nhất là góc trên-trái, thực ra là nền hoặc đồ vật khác, không thuộc bát. Polygon loại phần thừa đó, nên tính được diện tích vật chính xác hơn và tách được hai vật chồng lên nhau.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là tên gọi riêng của từng vật trong một ảnh và một lần chạy, ví dụ kitchen-002 là vật thứ hai của ảnh kitchen. Reviewer dùng nó để chỉ đúng một mask khi ghi chú (“sửa biên của kitchen-002”).
- Không phải class ID: class_id = 45 là số chung của mọi cái bát. Trong ảnh có nhiều bát (kitchen-002, 003, 006, 010), mỗi bát có instance_id riêng nhưng cùng class_id.
- Không phải tracking ID: tracking ID nối cùng một vật qua nhiều khung hình của video. kitchen-002 chỉ có nghĩa trong ảnh này. Nó không cho biết vật đó là cái nào ở ảnh khác.
- Nhìn file, số thứ tự có vẻ xếp theo score giảm dần, nên đổi ngưỡng hoặc đổi model thì số có thể đổi theo. Vì vậy nên ghi kèm sample_id khi trích dẫn.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Biên mask đi sát mép phần nhìn thấy của vật, không lấn ra nền hay sang vật khác. Mỗi pixel chỉ thuộc một vật. Khi vật bị che hoặc chạm vật khác, chỉ tô phần nhìn thấy và cắt biên ngay tại đường tiếp xúc.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Chỗ rõ nhất là góc dưới-trái, nơi dining table (0.58), oven (0.49), person (0.43) và vài bowl chồng lên nhau. - Từ JSON, box của table [0.97, 234.74, 345.86, 423.87] và box của oven [0.25, 188.95, 194.35, 294.62] giao nhau ở khoảng y từ 235 đến 295. Polygon của table có 598 điểm và nhảy lung tung, ví dụ sau [102, 287] là [22, 267] rồi [137, 265], nên biên chưa đáng tin.
- Cách xử lý:
    + Dùng guideline cho chuyện chồng lấn nói chung: bát nằm trên mặt bàn thì mask bát cắt khỏi mask bàn, biên theo đường tiếp xúc. Quy tắc ở câu 4 đủ để hai người cùng làm ra kết quả giống nhau.
    + Escalate lên Lab Coach cho phần guideline không quyết được: person 0.43 ở mép trái chỉ thấy một phần tay, và ranh giới oven với mặt bàn bị bột phủ khó nhìn. Hai chỗ này cần ai đó quyết là có gán nhãn hay không và vẽ biên theo cách nào. Sau khi có quyết định, nên ghi thêm vào guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn (hoặc một tập nhãn) cho cả ảnh, lấy từ bảng lớp ImageNet-1K. Không có vị trí, không có box.  |kitchen: top-1 là gong 0.42, rồi dining_table 0.07, restaurant 0.07, lumbermill 0.05, bakery 0.03. Điểm top-1 thấp và top-5 không có lớp nào là “bếp”, nên đây là ảnh khó quyết định. traffic: top-1 cab 0.51, trong khi detection thấy bus và car, nên nhãn ảnh và nhãn vật không khớp nhau. dining: restaurant 0.79 rất rõ, rank 2 chỉ 0.04.  |Xem cả ảnh, chọn một nhãn theo quy tắc “chủ thể chính” của guideline. Nếu không lớp nào phù hợp thì làm theo quy tắc cho ảnh không khớp lớp, đừng chọn bừa lớp gần nhất.  |Nhãn có đúng với cả ảnh không, các ảnh có điểm top-1 thấp như kitchen có được xử lý đúng quy tắc không.  |
| Phát hiện vật thể |Mỗi vật một bản ghi: class_name + bbox_xyxy (pixel). Đơn vị nhãn là một box quanh một vật.  |person 0.61, box [0.12, 263.18, 61.55, 310.78] sát mép trái, chỉ thấy phần tay (vật bị cắt mép). oven 0.69, box [0.11, 187.54, 195.96, 292.81] chồng lên bàn bột. cup 0.45 và 0.38 nằm sát nhau.  |Vẽ box ôm sát phần nhìn thấy, gán đúng lớp, không vẽ hai box cho một vật. Xử lý cắt mép theo guideline.  |Box có sát không, có đúng lớp không, có thiếu vật, thừa box, hay box trùng không.  |
| Instance segmentation |Mỗi vật một bản ghi: instance_id + class_name + polygon_xy. Đơn vị nhãn là một mask (viền) của một vật, tách riêng từng vật cùng lớp.  |dining table 0.58 (kitchen-005): polygon 598 điểm, có chỗ nhảy bất thường (sau [102, 287] là [22, 267]). spoon 0.49 (kitchen-008) có điểm lặp [534, 51], [534, 51]. potted plant 0.63 (kitchen-004) là bó lá treo, không thấy chậu, có thể sai lớp. Cùng một cái bát ở [155, 169, 182, 184], detection cho 0.70 còn segmentation cho 0.37.  |Vẽ biên sát phần nhìn thấy. Chỗ hai vật chạm nhau thì cắt tại đường tiếp xúc, mỗi pixel chỉ thuộc một vật.  |Biên có lấn sang vật khác không, polygon có điểm nhảy, điểm lặp hay lỗ không, các bát cạnh nhau có tách thành các instance_id riêng không.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ làm việc với các ảnh mẫu và file đầu ra của lab (traffic, kitchen, dining cùng các file JSON/PNG), và không đưa ảnh hay dữ liệu nào khác ra ngoài môi trường lab hoặc lên công cụ bên ngoài.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Tôi sẽ dừng xử lý và báo cho Lab Coach, người nhận các trường hợp escalation trong các câu hỏi trước của bạn. - Khi báo, tôi ghi rõ tên file hoặc sample_id, thấy gì bất thường (ví dụ ảnh có mặt người rõ nét, biển số, thông tin cá nhân, hoặc file không thuộc 3 ảnh mẫu) và vì sao tôi nghĩ nó ngoài phạm vi. Trong lúc chờ, tôi không chép, không chia sẻ và không xử lý tiếp dữ liệu đó.
- Nếu lab của bạn chỉ định một người hoặc kênh khác cho việc này (ví dụ người phụ trách dữ liệu), hãy thay bằng người đó.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
