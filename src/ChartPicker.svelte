<script>
    import { fade, fly } from 'svelte/transition';
    let visible = true;

    function turnOff(){
        visible = False;
    }
</script>

{#if visible}
    <div in:fly="{{duration: 1000 }}" out:fade>
        <button>
            Basic Tree
        </button>
        <button>
            Histogram Selector
        </button>
        <button on:click={turnOff}>
            x
        </button>
    </div>
{/if}
