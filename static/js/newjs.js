$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {

        if (input.files && input.files[0]) {

            var reader = new FileReader();

            reader.onload = function (e) {

                $('#imagePreview').attr('src', e.target.result);

            }

            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {

        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').html('');
        $('#result').hide();

        $('#confidence-bar').css('width', '0%');
        $('#confidence-bar').text('0%');

        readURL(this);
    });

    $('#btn-predict').click(function () {

        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({

            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,

            success: function (data) {

                $('.loader').hide();

                $('#result').fadeIn(600);

                if (data.result === "Tumor Detected") {

                    $('#result').html(
                        '<span style="color:red;font-size:28px;font-weight:bold;">🛑 ' +
                        data.result +
                        '</span><br><br><b>Confidence:</b> ' +
                        data.confidence + '%'
                    );

                } else {

                    $('#result').html(
                        '<span style="color:green;font-size:28px;font-weight:bold;">✅ ' +
                        data.result +
                        '</span><br><br><b>Confidence:</b> ' +
                        data.confidence + '%'
                    );
                }

                $('#confidence-bar').css(
                    'width',
                    data.confidence + '%'
                );

                $('#confidence-bar').text(
                    data.confidence + '%'
                );

                $('#btn-predict').show();

                console.log('Success!');
            }

        });

    });

});