<script>
    import ChartPicker from './ChartPicker.svelte';
    let numbers = [];
    function handleClick() {
		numbers = numbers.concat([numbers.length + 1]);
    }
</script>

<ul>
    {#each numbers as n}
        <ChartPicker></ChartPicker>
    {/each}
</ul>
<button on:click={handleClick}>+</button>
