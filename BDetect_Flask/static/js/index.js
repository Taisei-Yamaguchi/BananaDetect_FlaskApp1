// index.js

$(document).ready(function () {
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

                // レスポンスから画像のURLを取得
                var imageUrl =response; // レスポンスは画像のURLを含んでいると仮定
                
                // ランダムなクエリ文字列を生成
                var randomQueryString = Math.random().toString(36).substring(7);

                // 絶対パスを生成して画像を表示　クエリ文字を設定し直すことでキャッシュを無効化できる。
                var absoluteImageUrl = window.location.origin + imageUrl + '?' + randomQueryString;
                // 指定した要素に新しいHTMLコンテンツを追加
                $('.result-image').empty().append(`
                <h2>Detect Image</h2>
                <img src=${absoluteImageUrl} alt="Detect Image" width='80%'>
                `);

            },
            error: function (xhr, status, error) {
                console.error('画像のアップロード中にエラーが発生しました');
                console.error('ステータスコード:', status);
                console.error('エラーメッセージ:', error);
            },
        });
    });
});
