jq '[.cards[] | {colorIdentity: .colorIdentity, 
                colors: .colors, 
		convertedManaCost: .convertedManaCost, 
		manaCost: .manaCost, 
		name: .name, 
		rarity: .rarity, 
		scryfallId: .scryfallId,
		type: .type, 
		uuid: .uuid}]'
