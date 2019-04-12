jq '[.cards[] | {colorIdentity: .colorIdentity, 
                colors: .colors, 
		convertedManaCost: .convertedManaCost, 
		manaCost: .manaCost, 
		name: .name, 
		rarity: .rarity, 
		type: .type, 
		uuid: .uuid}]'
