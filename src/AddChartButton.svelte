<script>
    import ChartPicker from './ChartPicker.svelte';

    export let charts;

    let i = 0;
    function handleClick() {
        i++;
        charts = charts.concat([{type: "chartpicker", id: i}]);
        console.log(charts);
    }

    function deleteChart(id) {
		charts = charts.filter(t => t.id !== id);
	}
</script>

<ul>
    {#each charts as c}
        <ChartPicker chartInfo={c} on:delete={() => deleteChart(c.id)}></ChartPicker>
    {/each}
</ul>
<button on:click={handleClick}>+</button>
