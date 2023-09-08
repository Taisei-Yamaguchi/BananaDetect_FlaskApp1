//use jquery
$(document).ready(function () {
    //event when upload image
    $('#upload-form').submit(function (event) {
        event.preventDefault();
        var formData = new FormData(this);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                console.log('画像のアップロードが成功しました');
                console.log('レスポンス:', response);

                // レスポンスからdata取得
                var imageUrl =response.image_path; // レスポンスは画像のURLを含んでいると仮定
                var predictBrix=response.predict_brix;

                // ランダムなクエリ文字列を生成
                var randomQueryString = Math.random().toString(36).substring(7);
                // 絶対パスを生成して画像を表示　クエリ文字を設定し直すことでキャッシュを無効化できる。
                var absoluteImageUrl = window.location.origin + imageUrl + '?' + randomQueryString;

                var predictMes='';
                if(predictBrix>20){
                    predictMes='ちょー甘い';
                }else if(predictBrix>10){
                    predictMes='ほんのり甘い';
                }else{
                    predictMes='食べどきではない';
                }

                // 指定した要素に新しいHTMLコンテンツを追加
                $('.result-image').empty().append(`
                <h2>Detect Image</h2>
                <img src=${absoluteImageUrl} alt="Detect Image" width='70%'>
                `);

                // 表示中の砂時計のアニメーションを追加
                $('.predict-brix').html('<strong>予測中...</strong><div class="loading-box"><div class="loading-brick"></div></div>');

                // 9秒後に predictBrix の表示を追加
                setTimeout(function () {
                    $('.predict-brix').empty().append(`
                    <p><strong>予測糖度: ${predictBrix}</strong></p>
                    <p><strong>評価: ${predictMes}</strong></p>
                    `);
                }, 9000);

            },
            error: function (xhr, status, error) {
                console.error('画像のアップロード中にエラーが発生しました');
                console.error('ステータスコード:', status);
                console.error('エラーメッセージ:', error);
            },
        });
    });
});
