$('#uploadForm').submit(function(event) {
    event.preventDefault();
    $('#uploadForm').hide();
    $('#progressBarContainer').show();
    $('#progress').css('width', '0%');
    $('#progressText').text('Scanning for S... this could take a while');
    var formData = new FormData(this);

    var simulateProgress = 0;
    var interval = setInterval(function(){
        simulateProgress += 1;
        $('#progress').css('width', simulateProgress + '%');
        $('#progressText').text(simulateProgress + '%');
        if (simulateProgress >= 95){
            clearInterval(interval);
        }
    }, 100);

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
            if (simulateProgress < 95) {
                clearInterval(interval); 
                var fastProgress = setInterval(function() {
                    simulateProgress += 5; 
                    $('#progress').css('width', simulateProgress + '%');
                    $('#progressText').text(simulateProgress + '% Scanning...');

                    if (simulateProgress >= 100) {
                        clearInterval(fastProgress); 
                        $('#progressText').text('100% Complete!');
                    }
                }, 50);  
            } else {
                $('#progress').css('width', '100%');
                $('#progressText').text('100% Complete!');
            }
            setTimeout(function() {
                $('#progressBarContainer').hide(); 
                $('#processedImage').attr('src', imgURL);
                $('#result').css('display', 'flex').hide().fadeIn(500);
                $('html, body').animate({
                    scrollTop: $('#result').offset().top
                }, 1000);
            }, 1000); 
        },
        error: function() {
            $('#progressText').text('Error occurred during scan.');
            $('#progress').css('background-color', 'red');
        }
    });
});


$('#closeBtn').click(function() {
    $('#result').fadeOut(500); 
    $('#uploadForm').show();
});

$('#result').click(function(event) {
    if (!$(event.target).is('#processedImage') && !$(event.target).is('#closeBtn')) {
        $('#result').fadeOut(500);
        $('#uploadForm').show();
    }
});