<script>
    let numbers = [1, 2, 3];
    function handleClick() {
		numbers = numbers.concat([numbers.length + 1]);
	}
</script>

<ul>
    {#each numbers as n}
        <li>{n}</li>
    {/each}
</ul>
<button on:click={handleClick}>+</button>
