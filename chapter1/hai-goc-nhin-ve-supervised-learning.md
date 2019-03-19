# Hai góc nhìn về Supervised Learning

Ở bài này mình đưa ra một định nghĩa sơ lược. Định nghĩa này không hoàn toàn chính xác nhưng nó giúp ta hiểu một cách khái quát nhất về supervised learning. Dần dần trong các bài học sau chúng ta sẽ cải tiến để hoàn chỉnh định nghĩa này.

### Góc nhìn thứ nhất: ước lượng hàm số

Tiếp cận supervised learning thực chất không đòi hỏi quá nhiều kiến thức cao siêu. Nó có thể được quy về thành một bài toán tối thiểu hàm số cơ bản trong toán học.

Bạn hãy tưởng tượng rằng đang chơi một trò chơi với thiên nhiên \(là một sức mạnh vô hình điều khiển mọi sự việc của vũ trụ\). Đầu tiên, thiên nhiên viết ra một hàm bí ẩn $$f^*$$ nào đó. Sau đó thiên nhiên đưa vào hàm số này một loạt các observation $$x^{(1)}, \cdots, x^{(N)}$$ để tạo ra một loạt các label $$y^{(1)} = f^*(x^{(1)}), \cdots, y^{(N)} = f^*(x^{(N)})$$ tương ứng. Sau đó, thiên nhiên đem giấu hàm $$f^*$$ đi và chỉ chừa lại các cặp observation-label $$(x^{(i)}, y^{(i)})$$ cho chúng ta nhìn thấy. Nhiệm vụ của chúng ta là khôi phục lại được hàm $$f^*$$ bằng một model $$f_w$$ một cách chính xác nhất có thể. Ta gọi đây là **trò chơi supervised learning**.

Mọi bài toán supervised learning đều có thể được định nghĩa theo cách này. Ví dụ trong bài toán phân loại văn bản, observation $$x$$ có thể là một văn bản, label $$y$$ là chủ đề của văn bản đó, còn $$f^*$$ là một chuyên gia đọc văn bản và tìm ra chủ đề của chúng. Mục tiêu của machine learning là tạo ra một chương trình máy tính $$f_w$$ để làm công việc này tốt như là chuyên gia $$f^*$$ kia.

**Q1**: Giả sử bằng cách nào đó ta biết được rằng $$x = (x_1, x_2)$$ \(một vector 2 chiều\) và $$y = f^*(x) = ax_1^2 + bx_2 + c$$. Cần bao nhiêu cặp dữ liệu $$(x, y)$$ để có thể tìm ra parameter $$(a, b, c)$$?

**A1**: Với mỗi cặp $$(x, y)$$ ta xây dựng được một phương trình $$y = ax_1^2 + bx_2 + c$$. Vì có 3 ẩn nên ta cần chỉ cần 3 phương trình là giải ra được tham số \(nếu tồn tại\). Tức là cần 3 điểm dữ liệu.

Trong thực tế thì mọi chuyện không đơn giản như vậy. Supervised learning đối mặt với nghịch lý sau đây: vì ta không thể nào giao tiếp được với tự nhiên, nên sẽ không bao giờ biết được $$f^{*}$$ có dạng như thế nào. Vì thế, dù ta có đưa ra một model $$f_w$$ để ước lượng $$f^{*}$$, cũng không ai biết $$f^{*}$$ là gì để cho biết là ta đang đúng hay sai. Nói cách khác, supervised learning là một trò chơi dự đoán mà không ai biết đáp án đúng là gì.

Đọc đến đây các bạn đừng nản lòng. Tuy là nghe có phần không tưởng, nhưng không riêng gì machine learning, các ngành khoa học cơ bản khác cũng chơi những trò chơi tương tự. Các bạn có nghĩ rằng Einstein nằm mơ thấy thiên nhiên thì thầm vào tai mình công thức $$E = mc^2$$?

Cho đến giờ, người ta vẫn phải làm thí nghiệm trong thực tế để kiểm chứng lại các lý thuyết của Einstein cho đến khi nó sai thì thôi. Trong supervised learning, ta cũng làm một điều tương tự như vậy: _kiểm chứng sự đúng đắn của một model bằng thực nghiệm_. Cho dù không biết hình dạng của $$f^{*}$$ ra sao, thì vẫn còn đó các cặp observation-label sinh ra từ hàm này. Ta sẽ đánh giá độ tốt của một model trên các dữ liệu thực tế này.

Ví dụ, nếu nhận được 100 cặp observation-label, ta chỉ dùng khoảng 80 cặp để xây dựng ra $$f_w$$. Còn lại 20 cặp, ta giấu không cho model nhìn thấy lúc nó học. Sau khi model học từ 80 cặp dữ liệu, ta sẽ cho các observation của 20 cặp dữ liệu còn lại vào model để tạo ra các label dự đoán, rồi so sánh chúng với các label thật \(do $$f^*$$ sinh ra\). 80 cặp được dùng để xây dựng ra model gọi là **training set** \(tập huấn luyện\), còn 20 cặp dùng để đánh giá model gọi là **test set** \(tập kiểm tra\). Tương ứng, supervised learning cũng được chia thành hai quá trình: **train** \(lúc model học\) và **test** \(lúc đánh giá model\).

![](http://khanhxnguyen.com/wp-content/uploads/2016/05/ML101-accuracy.png)

**Q2**: Tại sao không dùng tất cả dữ liệu để train rồi test trên đó luôn?

**A2**: Trong machine learning, có một nguyên tắc vô cùng, vô cùng quan trọng cần nhớ: **đó là quá trình train và test phải độc lập với nhau! Dữ liệu được dùng để test model phải không được model nhìn thấy lúc train.** Có rất nhiều cách để vi phạm nguyên tắc này, và điều dẫn đến một hậu quả "thảm khốc", **overfitting**. Mình sẽ giải thích về hiện tượng này trong một dịp khác. Nói nôm na là model của bạn sẽ biến thành một con vẹt, chẳng học được gì khác ngoài việc lặp lại những gì nó đã nhìn thấy. Vì thế, bạn phải chia dữ liệu ra thành training set và test set, và phải làm điều này trước khi train model.

Tuy đã làm cho trò chơi supervised learning trở nên hợp lệ, ta vẫn chưa thể chơi được. Có hai vấn đề phát sinh, đó là:

1. Thế nào là một model "tốt" trên test set? 2. Làm sao để tìm ra được một model "tốt" từ training set?

Vấn đề thứ nhất gọi là **evaluation problem** và vấn đề thứ hai gọi là **training problem**. _Giải một bài toán machine learning tức là bạn đi tìm lời giải cụ thể cho hai vấn đề này_.

### Góc nhìn thứ hai: tối ưu hàm số

Để giải quyết hai vấn đề đã nêu, ta cần đến góc nhìn thứ hai.

Đầu tiên, ta tập trung vào evaluation problem: giả sử đã tìm được một model $$f_w$$, làm thế nào để đo độ tốt của nó trên test set bằng một con số cụ thể?

Một trong những cách đơn giản nhất đó là đếm xem nó đoán sai bao nhiêu label thật trên test set. Ta giả sử model bị phạt 1 điểm với mỗi lần label dự đoán khác với label thật. Số điểm bị phạt trung bình được gọi là **error rate** \(độ sai sót\) của model. Error rate là một số thực trong đoạn \[0, 1\]. Theo ngôn ngữ toán học, error rate trên một test set $$D_{test}$$ của model $$f_w$$ được tính như sau:


$$
e_{D_{test}} = \frac{1}{|D_{test}|} \sum_{(x, y) \in D_{test}} \mathbb{I}\{ f_w(x) \neq y \}
$$


trong đó:

* $$|D_{test}|$$ số lượng các cặp $$(x, y)$$ của test set \(kí hiệu \|S\| nghĩa là lực lượng của tập hợp S\).
* $$\mathbb{I}\{.\}$$ sẽ trả về 1 nếu logic trong dấu ngoặc nhọn là đúng, 0 nếu sai.

Nếu lập trình, pseudocode sẽ như thế này:

```
N = |D_{test}|
error_rate = 0
for i = 0 .. N - 1
  if (f(x[i]) != y[i]) error_rate += 1
error_rate = error_rate / N
```

**Error rate càng thấp, thì model càng tốt**. Nếu đoán đúng hết tất cả cặp dữ liệu, ta đạt được error rate "trong mơ", 0%. Nhưng nên nhớ đấy là kết quả được đo trên một test set hữu hạn. Kết quả này chỉ đưa ra được một chặn trên và chặn dưới cho kết quả trên tập vô hạn \(xem thêm về [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)\). Nói cách khác, nếu tự nhiên có gửi đến một test set mới để đánh giá $$f_w$$, thì chưa chắc ta có thể lặp lại được error rate giống như trên test set cũ. Error rate được đánh giá trên test set càng lớn thì càng đáng tin cậy.

Ta có thể định nghĩa độ tốt theo rất nhiều cách khác nữa. Tổng quát, độ tốt của model được xác định dựa trên giá trị của **evaluation function** trên test set:


$$
\mathcal{L}_{D_{test}} (f_w) = \frac{1}{|D_{test}|} \sum_{(x, y) \in D_{test}} L \left( f_w(x), y \right)
$$
 trong đó $$f_w$$ là model được test, hàm $$L$$ là một **loss function**.

Error rate là một dạng evaluation function thường dùng với loss function là $$L \left( f_w(x), y \right) \equiv \mathbb{I}\{ f(x) \neq y \}$$ \(kí hiệu $$\equiv$$ đọc là "được định nghĩa bằng nhau"\).

Đến đây, ta sẵn sàng chơi trò chơi supervised learning dưới góc nhìn như một bài toán tối ưu hàm số. Hãy quay lại trả lời hai vấn đề trong phần trước:

1. Thế nào là một model "tốt" trên test set? $$\rightarrow$$ model cho giá trị của evaluation function trên test set càng nhỏ thì càng tốt.

2. Làm sao để tìm ra được một model tốt từ training set? $$\rightarrow$$ tìm model cực tiểu hóa giá trị của evaluation function trên training set.

Cụ thể hơn, sau khi chọn được evaluation function, supervised learning có thể được gói gọn trong 2 bước sau:

1. **Train** \(huấn luyện\): tìm model $$f_w$$ tối thiểu hóa giá trị của evaluation function trên training set.
2. **Test** \(kiểm tra\): thông báo độ tốt của $$f_w$$ là gía trị của evaluation function trên test set.

**Lưu ý**: Ở đây ta sử dụng luôn evaluation fucntion để giải quyết training problem. Tuy nhiên, đây chỉ là giải pháp tạm thời. Lý do vì sao và giải pháp tốt hơn là gì sẽ được bàn đến trong bài sau.

