$('#uploadForm').submit(function(event) {
    event.preventDefault();
    var formData = new FormData(this);

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        xhr: function() {
            var xhr = new XMLHttpRequest();
            xhr.responseType = 'blob';
            return xhr;
        },
        success: function(response) {
            var imgURL = URL.createObjectURL(response);
            $('#processedImage').attr('src', imgURL);
            $('#result').css('display', 'flex').hide().fadeIn(500);
            $('html, body').animate({
                scrollTop: $('#result').offset().top
            }, 1000);
        }
    });
});

// Close button click event
$('#closeBtn').click(function() {
    $('#result').fadeOut(500); // Fade out the image container
});

// Close if clicking outside the image (on the background)
$('#result').click(function(event) {
    if (!$(event.target).is('#processedImage') && !$(event.target).is('#closeBtn')) {
        $('#result').fadeOut(500); // Fade out the image container
    }
});