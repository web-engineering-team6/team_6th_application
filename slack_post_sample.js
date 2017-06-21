$(function () {
    $('.slack-submit').on('click', function () {
    var url = 'https://slack.com/api/chat.postMessage';
        var data = {
            token: 'xoxp-190466053957-189718136128-201998499638-30f1441b2bfca899200698c2744db4e6',
            channel: '#random',
            username: 'yoshinobu-bot',
            text: 'Web-engineering-team6 First!'
        };

        $.ajax({
            type: 'GET',
            url: url,
            data: data,
            success: function (data) {
                alert( 'Can I post to Slack? :' + data.ok );
            }
        });
    });
});
