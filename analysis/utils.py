def get_message_length(message, padding_token):

	message[-1] = padding_token
	message[-2] = padding_token
	message[-3] = padding_token

	print(message)


	assert message[0] == padding_token
	
	pad_idx = 0

	for i in reversed(range(len(message))):
		if message[i] == padding_token:
			pad_idx = i
		else:
			break

	# we ignore the first token for statistical purposes
	if pad_idx == 0:
		return len(message) - 1
	else:
		return pad_idx - 1

