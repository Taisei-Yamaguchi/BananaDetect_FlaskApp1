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
                console.log('Success upload image');
                console.log('Response:', response);

                // get data from response
                var imageUrl =response.image_path; // image_path
                var predict=response.predict; //predict status

                // generate random query
                var randomQueryString = Math.random().toString(36).substring(7);
                // 絶対パスを生成して画像を表示　クエリ文字を設定し直すことでキャッシュを無効化できる!
                var absoluteImageUrl = window.location.origin + imageUrl + '?' + randomQueryString;
                // modelの結果が全て1になる原因は下の===を=にしていたから！！！
                var predictMes='';
                if(predict===0){
                    predictMes='It\'s edible. But it\'s good to wait a little..';
                }else if(predict===1){
                    predictMes='It should be sweet. Le\'s eat!';
                }else if(predict===2){
                    predictMes='Be careful. It\'s perioshble!';
                }

                // add new Element
                //image of Dtect Result
                $('.result-image').empty().append(`
                <h2>Detect Image</h2>
                <img src=${absoluteImageUrl} alt="Detect Image" width='70%'>
                `);

                // 表add animation
                $('.predict-brix').html('<strong>Predicting...</strong><div class="loading-box"><div class="loading-brick"></div></div>');

                // add Predict Result after 9 seconds.
                setTimeout(function () {
                    $('.predict-brix').empty().append(`
                    <p><strong>予測レベル: ${predict}</strong></p>
                    <p><strong>評価: ${predictMes}</strong></p>
                    `);
                }, 9000);

            },
            error: function (xhr, status, error) {
                console.error('some errors occured during uploading image.');
                console.error('StatusCode:', status);
                console.error('Error:', error);
            },
        });
    });
});
