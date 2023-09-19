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
                var predict=response.predict;

                // ランダムなクエリ文字列を生成
                var randomQueryString = Math.random().toString(36).substring(7);
                // 絶対パスを生成して画像を表示　クエリ文字を設定し直すことでキャッシュを無効化できる。
                var absoluteImageUrl = window.location.origin + imageUrl + '?' + randomQueryString;
                // modelの結果が全て1nいなる原因は下の===を=にしていたから！！！
                var predictMes='';
                if(predict===0){
                    predictMes='食べられるけどもう少し持ってもいいかも';
                }else if(predict===1){
                    predictMes='十分食べごろですね';
                }else if(predict===2){
                    predictMes='腐りつつあるから注意';
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
                    <p><strong>予測レベル: ${predict}</strong></p>
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
